"""Audit a finished run without calling any model.

  python3 -m promo_finder audit --run runs/gemini-batch --out runs/gemini-batch-audited

Reads the run's promos.json, candidates_all.jsonl, manifest.json and raw/ replies, then:
  1. re-validates every final promo with the current deterministic rules
  2. re-reads the critic's saved scores with key normalisation (no new critic call)
  3. raises human-review flags (review.py) and applies only metadata corrections the
     screenplay supports (e.g. a cliffhanger block that is never on screen -> the last beat that is)
  4. re-ranks by the corrected score, keeping the original rank and score next to it
  5. reconciles the run's counts (candidates, repairs, live calls vs cache hits)

The original run folder is never modified.
"""
import glob
import json
import os
from typing import Dict, List

from . import timecode
from .candidates import BlockRef, Candidate, Segment, validate
from .critic import apply_critique
from .report import segments_json
from .review import apply_fixes, review
from .screenplay import load_show

CACHED_SECONDS = 0.05   # raw replies that took less than this were served from the cache


def _cand_from_row(row: dict, promo: dict = None) -> Candidate:
    c = Candidate(
        id=row["internal_id"], origin=row["origin"],
        segments=[Segment(s["episode"], s["from"], s["to"]) for s in row["segments"]],
        hook=BlockRef(*row["hook"]), hook_description=(promo or {}).get("hook", {}).get("description", ""),
        cliffhanger=BlockRef(*row["cliffhanger"]),
        cliffhanger_description=(promo or {}).get("cliffhanger", {}).get("description", ""),
        why_it_works=(promo or {}).get("why_it_works", ""),
        self_scores=row.get("self_scores") or {},
    )
    c.public_id = row.get("public_id", "")
    return c


def reconcile(run_dir: str, doc: dict, manifest: dict, rows: List[dict]) -> Dict:
    raw = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(run_dir, "raw", "*.json")))
           if not f.endswith("_system_prompts.json")]
    per_model: Dict[str, int] = {}
    for r in raw:
        per_model[r["model"]] = per_model.get(r["model"], 0) + 1
    cached = sum(1 for r in raw if r.get("seconds", 0) < CACHED_SECONDS)
    initial = [r for r in rows if not r.get("repaired_from")]
    repairs = [r for r in rows if r.get("repaired_from")]
    out = {
        "initial_candidates": len(initial),
        "initial_failed_validation": sum(1 for r in initial if r["errors"]),
        "repair_outputs": len(repairs),
        "repair_outputs_valid": sum(1 for r in repairs if not r["errors"]),
        "total_candidate_records": len(rows),
        "sent_to_critic": manifest.get("sent_to_critic"),
        "final_promos": len(doc["ranked_promos"]),
        "rejected_records": len(doc["rejected_candidates"]),
        "recorded_responses": len(raw),
        "recorded_responses_per_model": per_model,
        "responses_served_from_cache_in_last_invocation": cached,
        "responses_made_live_in_last_invocation": len(raw) - cached,
        "manifest_live_calls": manifest.get("live_model_calls", manifest.get("model_calls")),
        "manifest_cache_hits": manifest.get("cache_hits"),
        "last_invocation_seconds": manifest.get("seconds"),
        "expected_requests_for_a_fresh_run": len(raw),
    }
    checks = []
    checks.append(("initial + repair outputs == total candidate records",
                   out["initial_candidates"] + out["repair_outputs"] == out["total_candidate_records"]))
    checks.append(("final + rejected == total candidate records",
                   out["final_promos"] + out["rejected_records"] == out["total_candidate_records"]))
    if out["manifest_live_calls"] is not None and out["manifest_cache_hits"] is not None:
        checks.append(("manifest live calls + cache hits == recorded responses",
                       out["manifest_live_calls"] + out["manifest_cache_hits"] == out["recorded_responses"]))
        checks.append(("raw timings agree with manifest cache hits",
                       out["responses_served_from_cache_in_last_invocation"] == out["manifest_cache_hits"]))
    if out["sent_to_critic"] is not None:
        checks.append(("sent to critic <= initially valid + valid repairs (exact duplicates are removed first)",
                       out["sent_to_critic"] <= (out["initial_candidates"] - out["initial_failed_validation"]
                                                 + out["repair_outputs_valid"])))
    out["checks"] = [{"check": c, "ok": bool(ok)} for c, ok in checks]
    return out


def run_audit(data_dir: str, run_dir: str, out_dir: str) -> dict:
    show = load_show(data_dir)
    with open(os.path.join(run_dir, "promos.json"), encoding="utf-8") as f:
        doc = json.load(f)
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(run_dir, "candidates_all.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    by_pub = {r.get("public_id"): r for r in rows}

    audited, changes = [], []
    for p in doc["ranked_promos"]:
        row = by_pub[p["promo_id"]]
        c = _cand_from_row(row, p)
        validate(show, c)
        before_errors = list(c.errors)
        apply_critique(c, dict(row.get("critique") or {}))
        rev = review(show, c, critic_saw_offscreen=True)
        log = []
        if rev["proposed_fixes"]:
            log = apply_fixes(c, rev["proposed_fixes"])
            validate(show, c)
            rev_after = review(show, c, critic_saw_offscreen=True)
            fixed_codes = {f["code"] for f in rev["flags"] if f["action"] == "repair"}
            for f in rev_after["flags"]:
                f["after_fix"] = True
            rev = {**rev_after, "fixed": [f for f in rev["flags"] if f["code"] in fixed_codes]}
        if p["total_score"] != c.total_score:
            log.append(f"total score {p['total_score']} -> {c.total_score} (critic scores re-read: "
                       f"{p['scoring']} -> {c.scores})")
        audited.append({"promo": p, "cand": c, "review": rev, "changes": log,
                        "validation_errors_before_fix": before_errors, "remaining_errors": list(c.errors)})
        if log:
            changes.append({"promo_id": p["promo_id"], "changes": log})

    audited.sort(key=lambda a: (-a["cand"].total_score, -a["cand"].scores.get("hook_strength", 0),
                                a["cand"].total_duration, a["promo"]["rank"]))

    ranked = []
    for rank, a in enumerate(audited, 1):
        p, c, rev = dict(a["promo"]), a["cand"], a["review"]
        p.update({
            "rank": rank,
            "original_rank": a["promo"]["rank"],
            "total_score": c.total_score,
            "original_total_score": a["promo"]["total_score"],
            "scoring": c.scores,
            "segments": segments_json(c),
            "total_duration_seconds": c.total_duration,
        })
        p["hook"] = {**p["hook"], "description": c.hook_description, "episode": c.hook.episode,
                     "timestamp": timecode.fmt(show[c.hook.episode].blocks[c.hook.block].start),
                     "offset_in_promo_seconds": c.hook_offset, "strength_score": c.scores["hook_strength"],
                     "source_line": _line(show, c.hook)}
        p["cliffhanger"] = {**p["cliffhanger"], "description": c.cliffhanger_description, "episode": c.cliffhanger.episode,
                            "timestamp": timecode.fmt(show[c.cliffhanger.episode].blocks[c.cliffhanger.block].start),
                            "strength_score": c.scores["cliffhanger_strength"], "source_line": _line(show, c.cliffhanger)}
        p["review"] = {"status": rev["status"], "suggested_action": rev["action"], "flags": rev["flags"],
                       "corrected": rev.get("fixed", [])}
        p["audit_changes"] = a["changes"]
        p["remaining_validation_errors"] = a["remaining_errors"]
        ranked.append(p)

    # Did any rejected candidate depend on a misread critic score?
    flipped = []
    for r in rows:
        if r.get("public_id", "").startswith("R") and r.get("critique") and not r["errors"]:
            c = _cand_from_row(r)
            validate(show, c)
            apply_critique(c, dict(r["critique"]))
            if (c.critique.get("missing_scores") or []) and not c.rejection_reasons:
                flipped.append(r["public_id"])

    rec = reconcile(run_dir, doc, manifest, rows)
    out_doc = {**doc, "ranked_promos": ranked,
               "audit": {"source_run": run_dir, "model_calls_made_by_audit": 0, "metadata_changes": changes,
                         "rejected_that_would_flip": flipped, "reconciliation": rec}}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "promos.json"), "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)
    _write_report(os.path.join(out_dir, "audit_report.md"), run_dir, ranked, changes, flipped, rec)
    return out_doc


def _line(show, ref: BlockRef) -> str:
    b = show[ref.episode].blocks[ref.block]
    return f"{b.speaker if b.kind == 'dialogue' else b.kind.upper()}: {b.text}"


def _write_report(path, run_dir, ranked, changes, flipped, rec):
    L = [f"# Audit of `{run_dir}`\n",
         "Produced by `python3 -m promo_finder audit` from the run's saved files. **No model was called.** "
         "The original run folder is unchanged.\n",
         "Flag kinds: **objective** = computed exactly from screenplay blocks; **approximate** = word overlap "
         "between a model's description and the clip (can misfire on paraphrases, so it only asks for a look); "
         "**subjective** = the LLM critic's judgement.\n"]
    counts = {}
    for p in ranked:
        counts[p["review"]["status"]] = counts.get(p["review"]["status"], 0) + 1
    L.append("## Summary\n")
    L.append(" · ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + f" (of {len(ranked)} promos)\n")
    L.append("## Metadata corrections applied (cut unchanged)\n")
    if changes:
        for c in changes:
            for line in c["changes"]:
                L.append(f"- **{c['promo_id']}**: {line}")
    else:
        L.append("- none")
    L.append("")
    L.append(f"Rejected candidates whose rejection depended on an unreadable critic score: "
             f"{', '.join(flipped) if flipped else 'none'}.\n")
    L.append("## Human review queue\n")
    L.append("| Rank (orig) | Promo | Cut | Status | Action | Flags | Audit corrections |\n|---|---|---|---|---|---|---|")
    for p in ranked:
        segs = " + ".join(f"E{s['episode']} {s['start_timestamp'][3:]}–{s['end_timestamp'][3:]}" for s in p["segments"])
        fl = "; ".join(f"{f['code']} ({f['kind']})" for f in p["review"]["flags"]) or "–"
        corr = "; ".join(f["code"] for f in p["review"].get("corrected", []))
        if any(ch.startswith("total score") for ch in p["audit_changes"]):
            corr = (corr + "; " if corr else "") + "score re-read"
        L.append(f"| {p['rank']} ({p['original_rank']}) | {p['promo_id']} | {segs} | {p['review']['status']} | "
                 f"{p['review']['suggested_action']} | {fl} | {corr or '–'} |")
    L.append("")
    L.append("## Flag details\n")
    for p in ranked:
        flags = p["review"]["flags"] + [dict(f, code=f["code"] + " (corrected)") for f in p["review"].get("corrected", [])]
        if not flags:
            continue
        L.append(f"### {p['promo_id']} (rank {p['rank']}, was {p['original_rank']}) · {p['total_score']}/100 · "
                 f"{p['review']['status']}\n")
        for f in flags:
            L.append(f"- **{f['code']}** [{f['kind']}, {f['action']}]: {f['detail']}")
            for b in f.get("blocks", [])[:4]:
                L.append(f"    - `{b}`")
        L.append("")
    L.append("## Count reconciliation\n")
    L.append("```json\n" + json.dumps({k: v for k, v in rec.items() if k != "checks"}, indent=1) + "\n```\n")
    for c in rec["checks"]:
        L.append(f"- {'✅' if c['ok'] else '❌'} {c['check']}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
