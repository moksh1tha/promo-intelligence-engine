"""Output writers: promos.json (assignment schema), report.md (human review)."""
import json
from typing import Dict, List

from . import timecode
from .candidates import Candidate, timeline
from .screenplay import Episode


def _block_text(show, ep, block_id) -> str:
    try:
        b = show[ep].blocks[block_id]
    except (KeyError, IndexError):
        return ""
    who = b.speaker if b.kind == "dialogue" else b.kind.upper()
    return f"{who}: {b.text}"


def _abs_ts(show, cand: Candidate, ref) -> str:
    try:
        return timecode.fmt(show[ref.episode].blocks[ref.block].start)
    except (KeyError, IndexError):
        return ""


def segments_json(cand: Candidate) -> List[dict]:
    return [{"episode": s.episode,
             "start_timestamp": timecode.fmt(s.start),
             "end_timestamp": timecode.fmt(s.end),
             "duration_seconds": s.duration,
             "block_range": [s.from_block, s.to_block]} for s in cand.segments]


def priority(rank: int) -> str:
    if rank <= 5:
        return "High"
    if rank <= 10:
        return "Medium"
    return "Low"


def promo_json(show, cand: Candidate, rank: int, rank_note: str) -> dict:
    crit = cand.critique or {}
    confirmed = crit.get("confirmed_flags")
    flags = {f: True for f in confirmed} if confirmed is not None else (crit.get("flags", {}) or {})
    return {
        "promo_id": cand.public_id,
        "rank": rank,
        "total_score": cand.total_score,
        "segments": segments_json(cand),
        "total_duration_seconds": cand.total_duration,
        "hook": {
            "description": cand.hook_description or crit.get("hook_assessment", ""),
            "timestamp": _abs_ts(show, cand, cand.hook),
            "episode": cand.hook.episode,
            "source_line": _block_text(show, cand.hook.episode, cand.hook.block),
            "offset_in_promo_seconds": cand.hook_offset,
            "strength_score": cand.scores.get("hook_strength", 0),
        },
        "standalone_context": crit.get("what_viewer_understands") or cand.context_note,
        "cliffhanger": {
            "description": cand.cliffhanger_description or crit.get("cliffhanger_assessment", ""),
            "timestamp": _abs_ts(show, cand, cand.cliffhanger),
            "episode": cand.cliffhanger.episode,
            "source_line": _block_text(show, cand.cliffhanger.episode, cand.cliffhanger.block),
            "open_question": crit.get("open_question", ""),
            "strength_score": cand.scores.get("cliffhanger_strength", 0),
        },
        "why_it_works": crit.get("why_it_works") or cand.why_it_works,
        "rank_rationale": rank_note,
        "scoring": cand.scores,
        "quality_checks": {
            "hook_within_first_3_seconds": cand.hook_offset is not None and cand.hook_offset <= 3.0 and not flags.get("weak_hook"),
            "standalone_context": not flags.get("missing_context"),
            "unresolved_ending": not flags.get("predictable_ending"),
            "generates_curiosity": not flags.get("low_curiosity"),
            "no_misleading_edit": not flags.get("misleading_edit"),
            "valid_duration": 20 <= cand.total_duration <= 60,
            "timestamps_verified": True,
        },
        "critic_confusions": crit.get("confusions", []),
        "critic_warnings": crit.get("warnings", []),
        "scout_self_scores": cand.self_scores,
        "origin": cand.origin,
        "recommended_publishing_priority": priority(rank),
        "borderline": bool(cand.borderline_reasons),
        "review": {"status": cand.review.get("status", "recommended"),
                   "suggested_action": cand.review.get("action", "accept"),
                   "flags": cand.review.get("flags", [])},
    }


def rejected_json(cand: Candidate) -> dict:
    reasons = list(cand.errors) + list(cand.rejection_reasons)
    crit = cand.critique or {}
    improvement = ""
    if cand.errors:
        if any("duration" in e for e in cand.errors):
            improvement = "Trim or extend the block range (or drop a segment) to land between 20 and 60 s."
        elif any("hook" in e for e in cand.errors):
            improvement = "Start the clip on the hook beat itself."
        elif any("cliffhanger" in e for e in cand.errors):
            improvement = "End the clip on the unresolved beat; cut the resolution that follows."
        else:
            improvement = "Re-select block ranges that exist in the episode."
    elif crit.get("confusions"):
        improvement = "Add a setup moment that answers: " + "; ".join(crit["confusions"][:2])
    elif any("hook" in r for r in reasons):
        improvement = "Open on a stronger beat (confrontation, reveal, strike) or prepend one."
    elif any("ending" in r or "cliff" in r for r in reasons):
        improvement = "End earlier, on the question or threat, before it resolves."
    return {
        "candidate_id": cand.public_id or cand.id,
        "episode": cand.segments[0].episode if cand.segments else None,
        "segments": segments_json(cand) if cand.segments and not any("outside episode" in e or "does not exist" in e for e in cand.errors) else
        [{"episode": s.episode, "block_range": [s.from_block, s.to_block]} for s in cand.segments],
        "total_duration_seconds": cand.total_duration if cand.segments and cand.segments[0].end else None,
        "stage_rejected": "validation" if cand.errors else ("dedup" if cand.duplicate_of else "critic"),
        "reason_for_rejection": "; ".join(reasons) or (f"near-duplicate of {cand.duplicate_of}" if cand.duplicate_of else ""),
        "potential_improvement": improvement,
        "scores": cand.scores or None,
        "origin": cand.origin,
        "repaired_from": cand.repaired_from,
    }


def write_markdown(path: str, show: Dict[int, Episode], doc: dict, ranked: List[Candidate], manifest: dict,
                   probes: List[dict], eval_result: dict):
    L = []
    L.append(f"# Promo report ({manifest['show_label']})\n")
    L.append(f"Run `{manifest['run_id']}` · model `{manifest['model']}` · prompts v{manifest['prompt_version']} · "
             f"{len(show)} episodes\n")
    s = doc["analysis_summary"]
    L.append(f"**{s['valid_promos']} promos passed** out of {s['total_candidates_identified']} candidates "
             f"({s['rejected_candidates']} rejected).\n")
    L.append("## Publishing plan\n")
    L.append(doc["ranking_explanation"] + "\n")
    L.append("## Ranked promos\n")
    for p, cand in zip(doc["ranked_promos"], ranked):
        segs = " → ".join(f"E{x['episode']} {x['start_timestamp'][3:]}–{x['end_timestamp'][3:]}" for x in p["segments"])
        L.append(f"### {p['rank']}. {p['promo_id']} · {p['total_score']}/100 · {p['total_duration_seconds']:.1f}s · "
                 f"{p['recommended_publishing_priority']} · review: {p['review']['status']}\n")
        L.append(f"- **Cut:** {segs}")
        L.append(f"- **Hook:** {p['hook']['description']}  \n  `{p['hook']['source_line'][:160]}`")
        L.append(f"- **Cliffhanger:** {p['cliffhanger']['description']}  \n  `{p['cliffhanger']['source_line'][:160]}`")
        L.append(f"- **Why it works:** {p['why_it_works']}")
        if p.get("rank_rationale"):
            L.append(f"- **Rank:** {p['rank_rationale']}")
        sc = p["scoring"]
        L.append(f"- Scores: hook {sc['hook_strength']}/25 · clarity {sc['standalone_clarity']}/20 · cliff "
                 f"{sc['cliffhanger_strength']}/25 · emotion {sc['emotional_intensity']}/15 · coherence "
                 f"{sc['narrative_coherence']}/10 · duration {sc['duration_and_boundary_quality']}/5")
        if p.get("critic_warnings"):
            L.append(f"- Critic warnings (flag raised, but the matching score is still acceptable): "
                     f"{'; '.join(p['critic_warnings'])}")
        if p.get("critic_confusions"):
            L.append(f"- Critic notes a new viewer may not understand: {'; '.join(p['critic_confusions'][:3])}")
        L.append("")
        L.append("<details><summary>Clip as the viewer sees it</summary>\n\n```")
        from .candidates import render_clip
        L.append(render_clip(show, cand))
        L.append("```\n</details>\n")

    L.append("## Human review queue\n")
    L.append("Promos the tool is unsure about. Flag kinds: objective = computed from screenplay blocks; "
             "approximate = word overlap between a description and the clip (may misfire on paraphrases); "
             "subjective = the LLM critic's judgement. Nothing here is auto-rejected.\n")
    queue = [p for p in doc["ranked_promos"] if p["review"]["flags"]]
    if not queue:
        L.append("- none\n")
    for p in queue:
        segs = " + ".join(f"E{x['episode']} {x['start_timestamp'][3:]}–{x['end_timestamp'][3:]}" for x in p["segments"])
        L.append(f"**#{p['rank']} {p['promo_id']}** ({segs}) · status `{p['review']['status']}` · suggested action "
                 f"`{p['review']['suggested_action']}`")
        for f in p["review"]["flags"]:
            L.append(f"- {f['code']} [{f['kind']}]: {f['detail']}")
            for b in f.get("blocks", [])[:3]:
                L.append(f"    - `{b}`")
        L.append("")

    L.append("## Quality control: where the tool made poor selections\n")
    rej = doc["rejected_candidates"]
    buckets = {}
    for r in rej:
        for reason in (r["reason_for_rejection"] or "unspecified").split("; "):
            key = _bucket(reason)
            buckets.setdefault(key, []).append(r["candidate_id"])
    L.append("| Failure type | Count | Examples |\n|---|---|---|")
    for k in sorted(buckets, key=lambda k: -len(buckets[k])):
        L.append(f"| {k} | {len(buckets[k])} | {', '.join(buckets[k][:6])} |")
    L.append("")
    over = manifest.get("overconfidence", [])
    if over:
        L.append("**Scout over-confidence** (scout rated itself ≥15 points higher than the blind critic):\n")
        for o in over[:12]:
            L.append(f"- {o['id']}: self {o['self']:.0f} vs critic {o['critic']}  ({o['origin']})")
        L.append("")
    if probes:
        L.append("### Critic calibration probes\n")
        L.append("Known-good and known-bad clips sent to the same critic to check it catches each failure type.\n")
        L.append("| Probe | Expected | Critic flags | Score | Correct verdict? | Right reason? |\n|---|---|---|---|---|---|")
        for p in probes:
            L.append(f"| {p['id']} | {p['expect']} | {', '.join(p['flags']) or '-'} | {p['score']} | "
                     f"{'yes' if p['ok'] else '**NO**'} | {'yes' if p.get('exact') else '**no**'} |")
        good = sum(p['ok'] for p in probes)
        L.append(f"\n{good}/{len(probes)} verdicts correct; "
                 f"{sum(bool(p.get('exact')) for p in probes)}/{len(probes)} for the expected reason.\n")
        L.append("")
    if eval_result:
        L.append("### Agreement with reference picks\n")
        L.append(f"{eval_result['matched']}/{eval_result['total']} reference promos are covered (≥50% time overlap) "
                 f"by a passing promo. Recall {eval_result['recall']:.0%}.\n")
        for r in eval_result["rows"]:
            L.append(f"- {r['id']} ({r['note']}): {'matched by ' + r['matched_by'] if r['matched_by'] else 'not found'}")
        L.append("")
    L.append("## Limitations\n")
    for lim in doc["limitations"]:
        L.append(f"- {lim}")
    L.append("\n## Run stats\n")
    L.append("```json\n" + json.dumps({k: v for k, v in manifest.items() if k not in ("overconfidence",)}, indent=1) + "\n```")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def _bucket(reason: str) -> str:
    r = reason.lower()
    if "duplicate" in r:
        return "near-duplicate of a better promo"
    if "duration" in r or "below 20" in r or "exceeds" in r:
        return "outside 20-60 s"
    if "hook starts" in r or "hook block" in r:
        return "hook not in first 3 s (timestamp check)"
    if "cliffhanger" in r and ("past" in r or "not in" in r or "block" in r):
        return "ends after the cliffhanger (timestamp check)"
    if "outside episode" in r or "does not exist" in r or "malformed" in r or "overlap" in r or "mid-line" in r:
        return "incorrect timestamps / block ids"
    if "weak hook" in r or "hook score" in r:
        return "weak hook"
    if "missing context" in r or "clarity" in r:
        return "missing context"
    if "predictable" in r or "cliffhanger score" in r:
        return "predictable ending"
    if "curiosity" in r:
        return "no curiosity"
    if "misleading" in r:
        return "misleading edit"
    return reason[:60]
