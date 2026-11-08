"""End-to-end run: parse -> scout -> validate/repair -> critic -> dedup -> rank -> report.

Every model call is logged verbatim (prompt + raw response) under
<out>/raw/, which is the "raw, unedited output" of a run.
"""
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import threading
import time
import traceback
from typing import Callable, Dict, List, Optional

from . import __version__, prompts
from .candidates import Candidate, overlap_seconds, render_clip, validate
from .critic import BatchLLMCritic, HeuristicCritic, LLMCritic, apply_critique
from .fixtures import load_fixture_file
from .llm import LLM, QuotaExhausted, ReplayMiss, extract_json
from .report import promo_json, rejected_json, write_markdown
from .memo import CritiqueMemo
from .repair import trim_to_rules
from .review import review
from .scout import BatchLLMScout, HeuristicScout, LLMScout
from .screenplay import Episode, load_show

DUP_OVERLAP = 0.5   # fraction of the shorter promo's runtime

# System prompts are written once to raw/_system_prompts.json; each raw call file names which one it used.
_SYSTEM_LABELS = {
    prompts.STRATEGIST_SYSTEM: "STRATEGIST_SYSTEM",
    prompts.CRITIC_SYSTEM: "CRITIC_SYSTEM",
    prompts.RANKING_SYSTEM: "RANKING_SYSTEM",
}



def dedup(passing: List[Candidate], reviews: Dict[int, dict]) -> List[Candidate]:
    """Drop near-duplicates from `passing` (already in ranking order). Of two overlapping clips the
    one without borderline scores, then without review flags, then ranked higher is kept; the
    dropped clip's duplicate_of names it. Returns the kept clips in ranking order."""
    rank = {id(c): i for i, c in enumerate(passing)}
    preference = sorted(passing, key=lambda c: (bool(c.borderline_reasons),
                                                reviews[id(c)]["status"] != "recommended", rank[id(c)]))
    kept: List[Candidate] = []
    for c in preference:
        dup = next((f for f in kept if overlap_seconds(c, f) > DUP_OVERLAP * min(c.total_duration, f.total_duration)), None)
        if dup:
            c.duplicate_of = dup.id
        else:
            kept.append(c)
    keep = {id(c) for c in kept}
    return [c for c in passing if id(c) in keep]


class Run:
    def __init__(self, data_dir: str, out_dir: str, llm: Optional[LLM], workers: int = 6,
                 max_per_episode: int = 3, max_arc_ideas: int = 8, episodes: Optional[List[int]] = None,
                 reference: Optional[str] = None, probes: Optional[str] = None, log: Callable[[str], None] = print,
                 critic_llm: Optional[LLM] = None, batch_episodes: int = 0, batch_critic: int = 0,
                 critic_policy: str = "borderline", gap_pass: bool = False, reuse_critiques: bool = True,
                 fallback_llm: Optional[LLM] = None, arc_planner_version: int = 2):
        """batch_episodes / batch_critic > 0 put several episodes / clips in one request (low-quota providers).
        critic_llm lets the critic use a different model (e.g. its own free-tier quota)."""
        self.critic_llm = critic_llm or llm
        self.critic_policy = critic_policy
        self.gap_pass = gap_pass
        self.arc_planner_version = arc_planner_version
        self.fallback_llm = fallback_llm
        self.memo = None
        self.reuse_critiques = reuse_critiques
        self.stats = {"failed_validation": 0, "auto_repaired": 0, "llm_repaired": 0, "critiques_reused": 0,
                      "borderline_kept": 0, "gap_episodes": [], "gap_candidates": 0}
        self.batch_episodes = batch_episodes
        self.batch_critic = batch_critic
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.llm = llm
        self.workers = workers
        self.max_per_episode = max_per_episode
        self.max_arc_ideas = max_arc_ideas
        self.episode_filter = episodes
        self.reference = reference
        self.probes_path = probes
        self.log = log
        self.raw_dir = os.path.join(out_dir, "raw")
        self._raw_names: set = set()  # raw file names written by this invocation
        self._raw_lock = threading.Lock()
        self.failures: List[dict] = []
        os.makedirs(self.raw_dir, exist_ok=True)
        with open(os.path.join(self.raw_dir, "_system_prompts.json"), "w", encoding="utf-8") as f:
            json.dump({v: k for k, v in _SYSTEM_LABELS.items()}, f, ensure_ascii=False, indent=1)

    # ---- model call with raw logging ------------------------------------
    def ask(self, stage: str, system: str, prompt: str, llm: Optional[LLM] = None) -> str:
        llm = llm or self.llm
        t0 = time.time()
        try:
            text = llm.complete(system, prompt)
        except (QuotaExhausted, ReplayMiss) as exc:
            # A replay miss falls back too, so a replay reproduces runs where the fallback answered.
            if self.fallback_llm is None or llm is self.fallback_llm:
                raise
            why = "quota exhausted" if isinstance(exc, QuotaExhausted) else "no cached reply"
            self.log(f"  {llm.describe()} {why}; trying fallback {self.fallback_llm.describe()} for {stage}")
            llm = self.fallback_llm
            text = llm.complete(system, prompt)
        # A stage name can recur in one invocation (the gap pass reuses repair/critic batch names):
        # give later calls a suffix so no raw response is overwritten.
        with self._raw_lock:
            name, n = stage, 1
            while name in self._raw_names:
                n += 1
                name = f"{stage}_{n}"
            self._raw_names.add(name)
        with open(os.path.join(self.raw_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump({"stage": name, "model": llm.describe(), "seconds": round(time.time() - t0, 2),
                       "system": _SYSTEM_LABELS.get(system, system),
                       "prompt": prompt, "response": text}, f, ensure_ascii=False, indent=1)
        return text

    def _parallel(self, tasks, label: str) -> Dict[str, object]:
        results = {}
        tasks = list(tasks)
        if not tasks:
            return results
        with cf.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futs = {pool.submit(fn): name for name, fn in tasks}
            done = 0
            for fut in cf.as_completed(futs):
                name = futs[fut]
                done += 1
                try:
                    results[name] = fut.result()
                except Exception as exc:  # keep going; record the failure
                    self.failures.append({"task": name, "error": f"{type(exc).__name__}: {exc}",
                                          "trace": traceback.format_exc(limit=3)})
                    self.log(f"  ! {name} failed: {type(exc).__name__}: {exc}")
                if done % 10 == 0 or done == len(tasks):
                    self.log(f"  {label}: {done}/{len(tasks)}")
        return results

    # ---- stages -----------------------------------------------------------
    def execute(self) -> dict:
        started = time.time()
        show_all = load_show(self.data_dir)
        if self.reuse_critiques and self.critic_llm is not None and getattr(self.critic_llm, "cache_dir", None):
            self.memo = CritiqueMemo(self.critic_llm.cache_dir, self.critic_llm.provider, self.critic_llm.model,
                                     self.critic_llm.effort, show_all)
        show = {n: e for n, e in show_all.items() if not self.episode_filter or n in self.episode_filter}
        self.log(f"Parsed {len(show)} episodes, {sum(len(e.blocks) for e in show.values())} blocks")
        for n, e in show.items():
            for w in e.warnings:
                self.log(f"  warning: episode {n} ({e.source_file}): {w}")
        parse_warnings = {n: e.warnings for n, e in show.items() if e.warnings}

        heuristic = self.llm is None
        # 1. scout
        if heuristic:
            scout = HeuristicScout()
            results = self._parallel(scout.episode_tasks(show), "heuristic scout")
            ideas = []
        else:
            if self.batch_episodes:
                scout = BatchLLMScout(self.ask, self.max_per_episode, self.max_arc_ideas, self.batch_episodes)
                scout.arc_planner_version = self.arc_planner_version
            else:
                scout = LLMScout(self.ask, self.max_per_episode, self.max_arc_ideas)
            tasks = list(scout.episode_tasks(show))
            ideas = []
            if len(show) > 1 and self.max_arc_ideas > 0:
                tasks.append(("arc_planner", lambda: scout.plan_arcs(show)))
            results = self._parallel(tasks, "scout")
            ideas = results.pop("arc_planner", []) or []
            if ideas:
                self.log(f"Arc planner proposed {len(ideas)} cross-episode ideas")
                results.update(self._parallel(scout.arc_tasks(show, ideas), "arc build"))
        cands: List[Candidate] = [c for name in sorted(results) for c in (results[name] or [])]
        self.log(f"Scouted {len(cands)} candidates")

        # 2-3. validate -> deterministic trim -> LLM repair -> critic
        self.stats = {"failed_validation": 0, "auto_repaired": 0, "llm_repaired": 0, "critiques_reused": 0,
                      "borderline_kept": 0, "gap_episodes": [], "gap_candidates": 0}
        self._seen_keys: Dict[tuple, Candidate] = {}
        all_cands, unique = self._process(show, cands, scout, heuristic)
        repaired = [c for c in all_cands if c.repaired_from]

        # 3b. optional gap pass: episodes with no surviving clip get one more, broader look
        if self.gap_pass and not heuristic and isinstance(scout, BatchLLMScout):
            covered = {e for c in unique if not c.rejection_reasons for e in c.episodes}
            gaps = [n for n in show if n not in covered]
            self.stats["gap_episodes"] = gaps
            if gaps:
                self.log(f"Gap pass: {len(gaps)} episodes have no surviving clip")
                res = self._parallel(scout.gap_tasks(show, gaps), "gap scout")
                gap_cands = [c for name in sorted(res) for c in (res[name] or [])]
                self.stats["gap_candidates"] = len(gap_cands)
                more_all, more_unique = self._process(show, gap_cands, scout, heuristic)
                all_cands += more_all
                unique += more_unique
                cands += gap_cands
        passing = sorted([c for c in unique if not c.rejection_reasons],
                         key=lambda c: (bool(c.borderline_reasons), -c.total_score,
                                        -c.scores.get("hook_strength", 0), c.total_duration))

        # 4. near-duplicate suppression. Of two overlapping clips keep the one an editor can use as is:
        # a clip with no review flag beats a flagged one, then the higher-ranked wins. (Before v0.3.3
        # only the score counted, so a flagged clip could displace a clean one.) The final order is
        # still the ranking order above.
        reviews = {id(c): review(show_all, c) for c in passing}   # local checks only, no model call
        final = dedup(passing, reviews)
        for i, c in enumerate(final, 1):
            c.public_id = f"P{i:03d}"
            c.review = reviews[id(c)]
        rejected = [c for c in all_cands if c not in final]
        for j, c in enumerate(rejected, 1):
            c.public_id = f"R{j:03d}"
        for c in rejected:
            if c.duplicate_of:
                target = next((x for x in all_cands if x.id == c.duplicate_of), None)
                if target is not None and target.public_id:
                    c.duplicate_of = target.public_id

        # 5. ranking narrative
        per_promo, plan = self._ranking_text(final)

        # 6. probes + reference eval
        probe_rows = self._run_probes(show_all, heuristic) if self.probes_path else []
        eval_result = self._evaluate(show_all, final) if self.reference else {}

        over = []
        for c in unique:
            if c.self_scores and c.scores:
                self_total = sum(c.self_scores.get(k, 0) for k in
                                 ("hook_strength", "standalone_clarity", "cliffhanger_strength", "emotional_intensity",
                                  "narrative_coherence"))
                crit_total = c.total_score - c.scores.get("duration_and_boundary_quality", 0)
                if self_total - crit_total >= 15:
                    over.append({"id": c.public_id, "origin": c.origin, "self": self_total, "critic": crit_total})
        over.sort(key=lambda o: o["critic"] - o["self"])

        doc = {
            "analysis_summary": {
                "episodes_analyzed": list(show),
                "total_candidates_identified": len(all_cands),
                "valid_promos": len(final),
                "rejected_candidates": len(rejected),
            },
            "ranked_promos": [promo_json(show_all, c, i, per_promo.get(c.public_id, "")) for i, c in enumerate(final, 1)],
            "rejected_candidates": [rejected_json(c) for c in rejected],
            "ranking_explanation": plan,
            "limitations": self._limitations(heuristic, parse_warnings),
        }
        manifest = {
            "run_id": os.path.basename(os.path.normpath(self.out_dir)),
            "show_label": os.path.basename(os.path.normpath(self.data_dir)),
            "started_utc": dt.datetime.utcfromtimestamp(started).isoformat() + "Z",
            "seconds": round(time.time() - started, 1),
            "model": self.llm.describe() if self.llm else "heuristic baseline (no model)",
            "critic_model": (self.critic_llm.describe() if self.critic_llm else "heuristic critic"),
            "batching": {"episodes_per_call": self.batch_episodes, "clips_per_critic_call": self.batch_critic},
            "prompt_version": prompts.PROMPT_VERSION,
            "code_version": __version__,
            "episodes": len(show),
            "candidates_scouted": len(cands),
            "failed_validation": self.stats["failed_validation"],
            "repaired": len([r for r in repaired if not r.errors]),
            "auto_repaired_no_model": self.stats["auto_repaired"],
            "llm_repaired": self.stats["llm_repaired"],
            "critiques_reused_per_clip": self.stats["critiques_reused"],
            "critic_policy": self.critic_policy,
            "borderline_kept_for_review": sum(1 for c in final if c.borderline_reasons),
            "gap_pass": {"episodes": self.stats["gap_episodes"], "candidates": self.stats["gap_candidates"]}
            if self.gap_pass else None,
            "sent_to_critic": len(unique),
            "passed_critic": len(passing),
            "final_promos": len(final),
            "arc_ideas": len(ideas),
            "live_model_calls": self._usage("calls"),
            "cache_hits": self._usage("cache_hits"),
            "recorded_responses": len([f for f in os.listdir(self.raw_dir) if f.endswith(".json")
                                       and not f.startswith("_")]),
            "counts_note": ("live_model_calls + cache_hits are for THIS invocation only; seconds is its wall time. "
                            "recorded_responses (files in raw/) is what a run with an empty cache would request."),
            "review_status_counts": _count(c.review.get("status") for c in final),
            "input_tokens": self._usage("input_tokens"),
            "output_tokens": self._usage("output_tokens"),
            "task_failures": self.failures,
            "scout_notes": getattr(scout, "notes", []),
            "parse_warnings": parse_warnings,
            "overconfidence": over,
        }
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "promos.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({**manifest, "probes": probe_rows, "reference_eval": eval_result}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.out_dir, "candidates_all.jsonl"), "w", encoding="utf-8") as f:
            for c in all_cands:
                f.write(json.dumps(_debug_row(c), ensure_ascii=False) + "\n")
        write_markdown(os.path.join(self.out_dir, "report.md"), show_all, doc, final, manifest, probe_rows, eval_result)
        self.log(f"Done: {len(final)} promos, {len(rejected)} rejected -> {self.out_dir}")
        return doc

    def _usage(self, attr: str) -> int:
        llms = {id(x): x for x in (self.llm, self.critic_llm, self.fallback_llm) if x is not None}
        return sum(getattr(x, attr, 0) for x in llms.values())

    def _ranking_text(self, final: List[Candidate]):
        if not final:
            return {}, "No candidate passed quality control."
        if self.llm is None:
            plan = ("Publish in score order. Heuristic baseline: scores come from keyword intensity, not "
                    "comprehension, so a human should review the top picks before publishing.")
            return {c.public_id: f"Score {c.total_score}." for c in final}, plan
        table = "\n".join(
            f"{c.public_id} | score {c.total_score} | eps {','.join(map(str, c.episodes))} | {c.total_duration:.0f}s | "
            f"hook {c.scores['hook_strength']}/25 clarity {c.scores['standalone_clarity']}/20 cliff "
            f"{c.scores['cliffhanger_strength']}/25 | {(c.critique or {}).get('why_it_works', c.why_it_works)}"
            for c in final)
        prompt = prompts.RANKING_TEMPLATE.format(promo_table=table)
        try:
            try:
                text = self.ask("ranking", prompts.RANKING_SYSTEM, prompt)
            except (QuotaExhausted, ReplayMiss):
                if self.critic_llm is None or self.critic_llm is self.llm:
                    raise
                self.log("  main model quota exhausted; writing the publishing plan with the critic model")
                text = self.ask("ranking", prompts.RANKING_SYSTEM, prompt, llm=self.critic_llm)
            data = extract_json(text)
            return data.get("per_promo", {}) or {}, data.get("publishing_plan", "")
        except Exception as exc:
            self.failures.append({"task": "ranking", "error": str(exc)})
            return {}, "Publish in score order (ranking narrative call failed)."

    def _process(self, show, cands: List[Candidate], scout, heuristic: bool):
        """Validate, repair (deterministic trim first, then one LLM attempt) and critique a batch of
        candidates. Returns (all candidate records incl. repairs, unique valid candidates)."""
        for c in cands:
            if not c.errors:
                validate(show, c)
        invalid = [c for c in cands if c.errors]
        self.stats["failed_validation"] += len(invalid)
        repaired: List[Candidate] = []
        still = []
        for c in invalid:
            t = trim_to_rules(show, c)
            if t is not None:
                repaired.append(t)
            else:
                still.append(c)
        self.stats["auto_repaired"] += len(repaired)
        if invalid:
            self.log(f"{len(invalid)} candidates failed validation; {len(repaired)} fixed by trimming edges "
                     f"(no model call), {len(still)} sent to LLM repair")
        if still and not heuristic:
            if isinstance(scout, BatchLLMScout):
                rep = self._parallel([("repair_batch", lambda: scout.repair_many(show, still))], "repair")
                llm_rep = list(rep.get("repair_batch") or [])
            else:
                rep = self._parallel([(f"repair:{c.id}", (lambda c=c: scout.repair(show, c))) for c in still
                                      if c.segments], "repair")
                llm_rep = [r for r in rep.values() if r is not None]
            self.stats["llm_repaired"] += sum(1 for r in llm_rep if not r.errors)
            repaired += llm_rep
        all_cands = cands + repaired
        unique = []
        for c in all_cands:
            if c.errors:
                continue
            key = tuple((s.episode, s.from_block, s.to_block) for s in c.segments)
            if key in self._seen_keys:
                c.duplicate_of = self._seen_keys[key].id
                continue
            self._seen_keys[key] = c
            unique.append(c)
        self.log(f"{len(unique)} unique valid candidates go to the critic")
        crit_results = self._critique(show, unique, heuristic)
        for c in unique:
            if c.id in crit_results:
                apply_critique(c, crit_results[c.id], policy=self.critic_policy)
            else:
                c.rejection_reasons = ["critic call failed"]
        return all_cands, unique

    def _critic_ask(self, stage, system, prompt):
        return self.ask(stage, system, prompt, llm=self.critic_llm)

    def _critique(self, show, cands: List[Candidate], heuristic: bool, stage: str = "critic") -> Dict[str, dict]:
        """Returns {candidate.id: critique}. One request per clip, or batches of clips."""
        if heuristic:
            critic = HeuristicCritic()
            return self._parallel([(c.id, (lambda c=c: critic.critique(show, c))) for c in cands], stage)
        if self.batch_critic:
            critic = BatchLLMCritic(self._critic_ask, per_call=self.batch_critic)
            size = self.batch_critic
            reused: Dict[str, dict] = {}
            if self.memo is not None:
                for c in cands:
                    r = self.memo.get(c)
                    if r is not None:
                        reused[c.id] = r
                self.stats["critiques_reused"] += len(reused) if stage == "critic" else 0
                if reused:
                    self.log(f"  {len(reused)} of {len(cands)} clips already reviewed by this critic model; reusing")
            todo = [c for c in cands if c.id not in reused]
            chunks = [todo[k:k + size] for k in range(0, len(todo), size)]
            res = self._parallel([(f"{stage}_batch{i:02d}", (lambda ch=ch, i=i: critic.critique_many(
                show, ch, stage=f"{stage}_batch{i:02d}_"))) for i, ch in enumerate(chunks, 1)], stage)
            merged: Dict[str, dict] = dict(reused)
            for part in res.values():
                merged.update(part or {})
            missing = [c.id for c in cands if c.id not in merged]
            if missing:
                self.log(f"  critic returned no review for {len(missing)} clip(s)")
            return merged
        critic = LLMCritic(self._critic_ask)
        return self._parallel([(c.id, (lambda c=c: critic.critique(show, c))) for c in cands], stage)

    def _run_probes(self, show, heuristic: bool) -> List[dict]:
        rows = []
        try:
            items = load_fixture_file(show, self.probes_path, origin="probe")
        except Exception as exc:
            self.failures.append({"task": "probes", "error": str(exc)})
            self.log(f"  probes skipped: {exc}")
            return rows
        results = self._critique(show, [c for c, _ in items], heuristic, stage="probe")
        for cand, item in items:
            crit = results.get(cand.id)
            if crit is None:
                continue
            apply_critique(cand, crit, policy=self.critic_policy)
            flags = [k for k, v in (crit.get("flags") or {}).items() if v is True]
            expect = item["expect"]
            verdict = "reject" if cand.rejection_reasons else ("review" if cand.borderline_reasons else "pass")
            rejected = verdict == "reject"
            # a bad clip is "caught" if it is rejected or sent to human review; a good one must pass
            ok = (verdict == "pass") if expect == "pass" else (verdict != "pass")
            exact = (verdict == "pass") if expect == "pass" else (expect in flags)
            rows.append({"id": cand.id, "expect": expect, "flags": flags, "score": cand.total_score,
                         "rejected": rejected, "verdict": verdict, "ok": ok, "exact": exact,
                         "reasons": cand.rejection_reasons, "note": item.get("note", "")})
        return rows

    def _evaluate(self, show, final: List[Candidate]) -> dict:
        try:
            refs = load_fixture_file(show, self.reference, origin="reference")
        except Exception as exc:
            self.failures.append({"task": "reference", "error": str(exc)})
            self.log(f"  reference skipped: {exc}")
            return {}
        in_scope = [r for r in refs if not self.episode_filter or all(e in self.episode_filter for e in r[0].episodes)]
        rows = []
        for ref, item in in_scope:
            best = None
            for c in final:
                if overlap_seconds(ref, c) >= 0.5 * ref.total_duration:
                    best = c.public_id
                    break
            rows.append({"id": ref.id, "note": item.get("note", ""), "matched_by": best})
        matched = sum(1 for r in rows if r["matched_by"])
        return {"total": len(rows), "matched": matched, "recall": matched / len(rows) if rows else 0.0, "rows": rows}

    def _limitations(self, heuristic: bool, parse_warnings) -> List[str]:
        lim = [
            "Works from screenplay text only: performance, framing, music and pacing of the final video are not seen. "
            "Timestamps are the screenplay's; verify against the rendered episode before cutting.",
            "Segments start/end on screenplay block boundaries; CAMERA shots that overlap a cut are trimmed, "
            "dialogue/ACTION/SFX never are.",
            "Scores are model judgements (a blind 'first-time viewer' critic), not measured CTR/retention. "
            "Use publish results to recalibrate thresholds.",
            "Cross-episode promos are proposed from loglines/continuity notes, so long-range links the notes do not "
            "mention can be missed.",
            "Spoilers are not penalised by the score; the publishing plan weighs spoiler risk qualitatively.",
        ]
        if heuristic:
            lim.insert(0, "HEURISTIC BASELINE RUN: no language model was used. Hooks/cliffhangers are found from "
                          "keywords, punctuation and episode structure; context and curiosity are not understood.")
        if parse_warnings:
            lim.append(f"Parser warnings in {len(parse_warnings)} episode(s); see manifest.json.")
        return lim


def _count(items) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def _debug_row(c: Candidate) -> dict:
    return {
        "internal_id": c.id, "public_id": c.public_id, "origin": c.origin,
        "segments": [{"episode": s.episode, "from": s.from_block, "to": s.to_block, "start": s.start, "end": s.end}
                     for s in c.segments],
        "duration": c.total_duration if c.segments else None,
        "hook": [c.hook.episode, c.hook.block], "cliffhanger": [c.cliffhanger.episode, c.cliffhanger.block],
        "errors": c.errors, "warnings": c.warnings, "rejection_reasons": c.rejection_reasons,
        "duplicate_of": c.duplicate_of, "repaired_from": c.repaired_from,
        "self_scores": c.self_scores, "scores": c.scores, "total_score": c.total_score, "critique": c.critique,
    }
