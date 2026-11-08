"""Stage 1 - candidate generation ("scouting").

LLMScout:
  * one call per episode  -> single-episode candidates
  * one arc-planner call over all loglines -> cross-episode ideas
  * one build call per idea with only the relevant episodes' scripts
  * one repair call for any candidate that fails deterministic validation

HeuristicScout: a keyword/structure baseline that needs no model. It exists so
the pipeline runs offline, so tests are deterministic, and as a baseline to
show what the LLM adds.
"""
import json
import re
from typing import Callable, Dict, List, Optional, Tuple

from . import prompts
from .candidates import BlockRef, Candidate, Segment, render_clip, segment_end, validate
from .llm import extract_json
from .screenplay import Block, Episode

Ask = Callable[[str, str, str], str]  # (stage_name, system, prompt) -> text


def _series_index(show: Dict[int, Episode], upto: Optional[int] = None) -> str:
    lines = []
    for n, ep in show.items():
        if upto is not None and n >= upto:
            break
        lines.append(f"Ep {n}: {ep.logline}")
    return "\n".join(lines) or "(this is the first episode)"


def parse_candidates(payload, origin: str) -> Tuple[List[Candidate], str]:
    """Turn model JSON into Candidate objects. Malformed entries become candidates with errors."""
    out = []
    skipped = ""
    if isinstance(payload, dict):
        skipped = payload.get("skipped_reason", "") or ""
        items = payload.get("candidates", [])
    else:
        items = payload
    for i, c in enumerate(items or []):
        try:
            segs = [Segment(int(s["episode"]), int(s["from"]), int(s["to"])) for s in c["segments"]]
            hook = c.get("hook", {})
            cliff = c.get("cliffhanger", {})
            cand = Candidate(
                id=f"{origin}#{i + 1}",
                origin=origin,
                segments=segs,
                hook=BlockRef(int(hook.get("episode", segs[0].episode)), int(hook.get("block", segs[0].from_block))),
                hook_description=hook.get("description", ""),
                cliffhanger=BlockRef(int(cliff.get("episode", segs[-1].episode)), int(cliff.get("block", segs[-1].to_block))),
                cliffhanger_description=cliff.get("description", ""),
                why_it_works=c.get("why_it_works", ""),
                context_note=c.get("standalone_context", ""),
                self_scores={k: float(v) for k, v in (c.get("self_scores") or {}).items() if _is_num(v)},
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            cand = Candidate(id=f"{origin}#{i + 1}", origin=origin, segments=[], hook=BlockRef(0, 0),
                             hook_description="", cliffhanger=BlockRef(0, 0), cliffhanger_description="",
                             why_it_works="", errors=[f"malformed candidate JSON: {exc}"])
        out.append(cand)
    return out, skipped


def _is_num(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def candidate_to_json(c: Candidate) -> str:
    return json.dumps({
        "segments": [{"episode": s.episode, "from": s.from_block, "to": s.to_block} for s in c.segments],
        "hook": {"episode": c.hook.episode, "block": c.hook.block, "description": c.hook_description},
        "cliffhanger": {"episode": c.cliffhanger.episode, "block": c.cliffhanger.block,
                        "description": c.cliffhanger_description},
        "standalone_context": c.context_note,
        "why_it_works": c.why_it_works,
    })


class LLMScout:
    def __init__(self, ask: Ask, max_per_episode: int = 3, max_arc_ideas: int = 8):
        self.ask = ask
        self.max_per_episode = max_per_episode
        self.max_arc_ideas = max_arc_ideas
        self.notes: List[str] = []

    # -- single-episode -------------------------------------------------
    def episode_tasks(self, show: Dict[int, Episode]):
        for n, ep in show.items():
            yield f"scout_ep{n:03d}", (lambda n=n, ep=ep: self._scout_episode(show, n, ep))

    def _scout_episode(self, show, n, ep) -> List[Candidate]:
        prompt = prompts.EPISODE_TEMPLATE.format(
            series_so_far=_series_index(show, upto=n),
            episode_text=ep.render(),
            max_candidates=self.max_per_episode,
            schema=prompts.CANDIDATE_SCHEMA,
        )
        text = self.ask(f"scout_ep{n:03d}", prompts.STRATEGIST_SYSTEM, prompt)
        cands, skipped = parse_candidates(extract_json(text), f"ep{n:03d}")
        if skipped and not cands:
            self.notes.append(f"episode {n}: no candidate - {skipped}")
        return cands

    # -- cross-episode ----------------------------------------------------
    arc_planner_version = 1

    def plan_arcs(self, show) -> List[dict]:
        if self.arc_planner_version >= 2:
            def edge(ep, last):
                beats = [b for b in ep.blocks if b.kind == "dialogue"] or [b for b in ep.blocks if b.kind != "camera"]
                if not beats:
                    return ""
                b = beats[-1] if last else beats[0]
                return f"{b.speaker}: {b.text}" if b.kind == "dialogue" else b.text
            index = "\n".join(
                f"Ep {n} ({ep.duration:.0f}s): {ep.logline}\n   continuity: {ep.continuity}\n"
                f"   opens: {edge(ep, False)[:140]}\n   ends: {edge(ep, True)[:140]}" for n, ep in show.items())
            prompt = prompts.ARC_PLANNER_TEMPLATE_V2.format(episode_index=index, max_ideas=self.max_arc_ideas)
        else:
            index = "\n".join(
                f"Ep {n} ({ep.duration:.0f}s): {ep.logline}\n   continuity: {ep.continuity}" for n, ep in show.items())
            prompt = prompts.ARC_PLANNER_TEMPLATE.format(episode_index=index, max_ideas=self.max_arc_ideas)
        text = self.ask("arc_planner", prompts.STRATEGIST_SYSTEM, prompt)
        ideas = extract_json(text).get("ideas", [])
        valid = []
        for idea in ideas[: self.max_arc_ideas]:
            eps = [int(e) for e in idea.get("episodes", []) if int(e) in show]
            if 2 <= len(eps) <= 3:
                idea["episodes"] = sorted(set(eps))
                valid.append(idea)
            else:
                self.notes.append(f"arc idea dropped (episodes {idea.get('episodes')} not 2-3 valid episodes)")
        return valid

    def arc_tasks(self, show, ideas):
        for i, idea in enumerate(ideas, 1):
            yield f"arc_build{i:02d}", (lambda i=i, idea=idea: self._build_arc(show, i, idea))

    def _build_arc(self, show, i, idea) -> List[Candidate]:
        prompt = prompts.ARC_BUILD_TEMPLATE.format(
            concept=idea.get("concept", ""), hook_idea=idea.get("hook_idea", ""),
            cliffhanger_idea=idea.get("cliffhanger_idea", ""),
            episodes_text="\n\n".join(show[e].render() for e in idea["episodes"]),
            schema=prompts.CANDIDATE_SCHEMA,
        )
        text = self.ask(f"arc_build{i:02d}", prompts.STRATEGIST_SYSTEM, prompt)
        cands, skipped = parse_candidates(extract_json(text), f"arc{i:02d}")
        if skipped and not cands:
            self.notes.append(f"arc idea {i} ({idea.get('concept', '')[:60]}): not built - {skipped}")
        return cands

    # -- repair -------------------------------------------------------------
    def repair(self, show, cand: Candidate) -> Optional[Candidate]:
        eps = sorted({s.episode for s in cand.segments if s.episode in show}) or []
        if not eps:
            return None
        resolved = ", ".join(
            f"E{s.episode} #{s.from_block}-#{s.to_block} = {s.start:.2f}-{s.end:.2f}s ({s.duration:.2f}s)" for s in cand.segments)
        prompt = prompts.REPAIR_TEMPLATE.format(
            errors="\n".join("- " + e for e in cand.errors), resolved=resolved,
            candidate_json=candidate_to_json(cand),
            episodes_text="\n\n".join(show[e].render() for e in eps),
            schema=prompts.CANDIDATE_SCHEMA,
        )
        safe = re.sub(r"[^A-Za-z0-9]+", "_", cand.id)
        text = self.ask(f"repair_{safe}", prompts.STRATEGIST_SYSTEM, prompt)
        fixed, _ = parse_candidates(extract_json(text), cand.origin)
        if not fixed:
            return None
        new = fixed[0]
        new.id = cand.id + "r"
        new.repaired_from = cand.id
        return validate(show, new)


class BatchLLMScout(LLMScout):
    """Same stages as LLMScout with several items per request (for low requests/day quotas)."""

    def __init__(self, ask: Ask, max_per_episode: int = 3, max_arc_ideas: int = 5, episodes_per_call: int = 10):
        super().__init__(ask, max_per_episode, max_arc_ideas)
        self.episodes_per_call = max(1, episodes_per_call)

    def episode_tasks(self, show):
        nums = list(show)
        for i in range(0, len(nums), self.episodes_per_call):
            chunk = nums[i:i + self.episodes_per_call]
            name = f"scout_ep{chunk[0]:03d}-{chunk[-1]:03d}"
            yield name, (lambda chunk=chunk, name=name: self._scout_batch(show, chunk, name))

    def _scout_batch(self, show, chunk, name) -> List[Candidate]:
        prompt = prompts.EPISODE_BATCH_TEMPLATE.format(
            series_so_far=_series_index(show, upto=chunk[0]),
            episodes_text="\n\n".join(show[n].render() for n in chunk),
            min_candidates=min(2, self.max_per_episode),
            max_candidates=self.max_per_episode,
            schema=prompts.CANDIDATE_SCHEMA,
        )
        cands, _ = parse_candidates(extract_json(self.ask(name, prompts.STRATEGIST_SYSTEM, prompt)), name)
        counters: Dict[int, int] = {}
        for c in cands:
            ep = c.segments[0].episode if c.segments else chunk[0]
            counters[ep] = counters.get(ep, 0) + 1
            c.origin = f"ep{ep:03d}"
            c.id = f"ep{ep:03d}#{counters[ep]}"
            if c.segments and ep not in chunk:
                c.warnings.append(f"scout returned episode {ep} from a batch of episodes {chunk[0]}-{chunk[-1]}")
        for n in chunk:
            if n not in counters:
                self.notes.append(f"episode {n}: no candidate")
        return cands

    def arc_tasks(self, show, ideas):
        if ideas:
            yield "arc_build_batch", (lambda: self._build_arcs(show, ideas))

    def gap_tasks(self, show, episodes: List[int]):
        for i in range(0, len(episodes), self.episodes_per_call):
            chunk = episodes[i:i + self.episodes_per_call]
            name = f"gap_scout_ep{chunk[0]:03d}-{chunk[-1]:03d}"
            yield name, (lambda chunk=chunk, name=name: self._gap_batch(show, chunk, name))

    def _gap_batch(self, show, chunk, name) -> List[Candidate]:
        prompt = prompts.GAP_BATCH_TEMPLATE.format(
            series_so_far=_series_index(show, upto=chunk[0]), moment_types=prompts.MOMENT_TYPES,
            episodes_text="\n\n".join(show[n].render() for n in chunk),
            max_candidates=self.max_per_episode, schema=prompts.CANDIDATE_SCHEMA)
        cands, _ = parse_candidates(extract_json(self.ask(name, prompts.STRATEGIST_SYSTEM, prompt)), name)
        counters: Dict[int, int] = {}
        for c in cands:
            ep = c.segments[0].episode if c.segments else chunk[0]
            counters[ep] = counters.get(ep, 0) + 1
            c.origin = f"gap-ep{ep:03d}"
            c.id = f"gap{ep:03d}#{counters[ep]}"
        return cands

    def _build_arcs(self, show, ideas) -> List[Candidate]:
        concepts = "\n".join(
            f"{i}. episodes {idea['episodes']}: {idea.get('concept', '')} | hook: {idea.get('hook_idea', '')} | "
            f"ends: {idea.get('cliffhanger_idea', '')}" for i, idea in enumerate(ideas, 1))
        eps = sorted({e for idea in ideas for e in idea["episodes"]})
        prompt = prompts.ARC_BUILD_BATCH_TEMPLATE.format(
            concepts=concepts, episodes_text="\n\n".join(show[e].render() for e in eps),
            schema=prompts.CANDIDATE_SCHEMA)
        payload = extract_json(self.ask("arc_build_batch", prompts.STRATEGIST_SYSTEM, prompt))
        items = payload.get("candidates", []) if isinstance(payload, dict) else payload
        out = []
        for j, item in enumerate(items or [], 1):
            try:
                idea_no = int(item.get("idea", 0)) if isinstance(item, dict) else 0
            except (TypeError, ValueError):
                idea_no = 0
            origin = f"arc{idea_no:02d}" if idea_no else "arc00"
            cands, _ = parse_candidates({"candidates": [item]}, origin)
            for c in cands:
                c.id = f"{origin}#{j}"
                out.append(c)
        built = {c.origin for c in out}
        for i, idea in enumerate(ideas, 1):
            if f"arc{i:02d}" not in built:
                self.notes.append(f"arc idea {i} ({idea.get('concept', '')[:60]}): not built")
        return out

    def repair_many(self, show, cands: List[Candidate], per_call: int = 25) -> List[Candidate]:
        out = []
        cands = [c for c in cands if c.segments and all(s.episode in show for s in c.segments)]
        for k in range(0, len(cands), per_call):
            chunk = cands[k:k + per_call]
            items = []
            for c in chunk:
                resolved = ", ".join(f"E{s.episode} #{s.from_block}-#{s.to_block} = {s.start:.2f}-{s.end:.2f}s "
                                     f"({s.duration:.2f}s)" for s in c.segments)
                items.append(f"ID {c.id}\nErrors:\n" + "\n".join("- " + e for e in c.errors) +
                             f"\nResolved: {resolved}\nCandidate: {candidate_to_json(c)}")
            eps = sorted({s.episode for c in chunk for s in c.segments})
            prompt = prompts.REPAIR_BATCH_TEMPLATE.format(
                items="\n\n".join(items), episodes_text="\n\n".join(show[e].render() for e in eps),
                schema=prompts.CANDIDATE_SCHEMA)
            payload = extract_json(self.ask(f"repair_batch{k // per_call + 1:02d}", prompts.STRATEGIST_SYSTEM, prompt))
            by_id = {c.id: c for c in chunk}
            for r in (payload.get("repairs", []) if isinstance(payload, dict) else []):
                orig = by_id.get(str(r.get("id")))
                cand_obj = r.get("candidate")
                # models sometimes wrap the fix in the list form {"candidates": [...]}; accept both
                if isinstance(cand_obj, dict) and isinstance(cand_obj.get("candidates"), list):
                    cand_obj = cand_obj["candidates"][0] if cand_obj["candidates"] else None
                if orig is None or not isinstance(cand_obj, dict):
                    continue
                fixed, _ = parse_candidates({"candidates": [cand_obj]}, orig.origin)
                if fixed:
                    new = fixed[0]
                    new.id = orig.id + "r"
                    new.repaired_from = orig.id
                    out.append(validate(show, new))
        return out


# ---------------------------------------------------------------------------
# Heuristic baseline
# ---------------------------------------------------------------------------
_TENSE_TONES = re.compile(
    r"scream|shout|furious|hysteric|desperat|terrif|enrag|roar|shriek|menac|threat|venom|cruel|panic|"
    r"sob|weep|horrif|shock|accus|outrag|gasp|agon|deadly|lethal|sinister|chill", re.I)
_TENSE_ACTION = re.compile(
    r"strike|strikes|slap|punch|drag|grab|shove|crash|explod|burst|collaps|blood|slam|kneel|fire|whip|"
    r"tackl|lung|seiz|chain|flung|fling|snatch|yank|convuls|fist|sword|iron", re.I)


def intensity(b: Block) -> int:
    if b.kind == "camera":
        return 1 if re.search(r"cut to black|freeze", b.text, re.I) else 0
    score = 0
    if b.kind == "dialogue":
        score += 2 * b.text.count("!") + (1 if "?" in b.text else 0)
        score += 2 if _TENSE_TONES.search(b.tone) else 0
    if b.kind in ("action", "sfx"):
        score += 3 if _TENSE_ACTION.search(b.text) else 0
        score += 1 if b.kind == "sfx" else 0
    return score


class HeuristicScout:
    """Picks, per episode, the window with the most intense opening and an
    unresolved-looking final beat (questions, threats, episode-final beats)."""

    def __init__(self, per_episode: int = 2, target_min: float = 25.0, target_max: float = 58.0):
        self.per_episode = per_episode
        self.target_min = target_min
        self.target_max = target_max
        self.notes: List[str] = []

    def episode_tasks(self, show):
        for n, ep in show.items():
            yield f"heuristic_ep{n:03d}", (lambda n=n, ep=ep: self._scout(show, n, ep))

    def _scout(self, show, n, ep: Episode) -> List[Candidate]:
        blocks = ep.blocks
        story = [b for b in blocks if b.kind != "camera"]
        if not story:
            return []
        last_story = story[-1].id
        scored = []
        for e in story:
            cliff = intensity(e) + (4 if e.id == last_story else 0) + (3 if "?" in e.text else 0)
            if cliff < 3:
                continue
            for s in blocks:
                if s.id > e.id or s.kind == "camera":
                    continue
                seg_end = segment_end(blocks[s.id: e.id + 1])
                dur = seg_end - s.start
                if not (self.target_min <= dur <= self.target_max):
                    continue
                opening = [b for b in blocks[s.id: e.id + 1] if b.start - s.start <= 3.0 and b.kind != "camera"]
                hook_b = max(opening, key=intensity)
                if seg_end - e.end > 4.0:
                    continue
                body = sum(intensity(b) for b in blocks[s.id: e.id + 1]) / max(1.0, dur / 10)
                score = 3 * intensity(hook_b) + 2 * cliff + body
                scored.append((score, s.id, e.id, hook_b.id))
        scored.sort(reverse=True)
        picked: List[Candidate] = []
        for score, s, e, h in scored:
            cand = Candidate(
                id=f"h{n:03d}#{len(picked) + 1}", origin=f"heuristic-ep{n:03d}",
                segments=[Segment(n, s, e)], hook=BlockRef(n, h),
                hook_description=_short(blocks[h]), cliffhanger=BlockRef(n, e),
                cliffhanger_description=_short(blocks[e]),
                why_it_works="Heuristic: intense opening line/action and an unresolved-looking final beat.",
            )
            validate(show, cand)
            if cand.errors:
                continue
            if any(_overlap_frac(cand, p) > 0.3 for p in picked):
                continue
            picked.append(cand)
            if len(picked) >= self.per_episode:
                break
        return picked


def _short(b: Block) -> str:
    who = b.speaker if b.kind == "dialogue" else b.kind.upper()
    return f"{who}: {b.text[:140]}"


def _overlap_frac(a: Candidate, b: Candidate) -> float:
    from .candidates import overlap_seconds
    shorter = min(a.total_duration, b.total_duration) or 1.0
    return overlap_seconds(a, b) / shorter


__all__ = ["LLMScout", "BatchLLMScout", "HeuristicScout", "parse_candidates", "render_clip", "intensity"]
