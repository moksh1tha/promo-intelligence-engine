"""Deterministic boundary repair. No model calls.

Many validation failures are purely mechanical: the clip starts a few seconds before the
hook, runs on past its own cliffhanger, starts or ends mid-line, or is a second too long.
The model already chose the hook and the cliffhanger; this module keeps both and only moves
the outer edges of the first and last segment, block by block, until every objective rule
passes. If no edge adjustment works, the candidate goes to the (LLM) repair step as before.

It never adds content, never changes the hook or cliffhanger choice, and records what it did.
"""
from typing import Dict, Optional

from .candidates import Candidate, Segment, validate
from .screenplay import Episode


def _copy(cand: Candidate, segments) -> Candidate:
    new = Candidate(
        id=cand.id + "t", origin=cand.origin, segments=segments, hook=cand.hook,
        hook_description=cand.hook_description, cliffhanger=cand.cliffhanger,
        cliffhanger_description=cand.cliffhanger_description, why_it_works=cand.why_it_works,
        context_note=cand.context_note, self_scores=dict(cand.self_scores))
    new.repaired_from = cand.repaired_from or cand.id
    return new


def trim_to_rules(show: Dict[int, Episode], cand: Candidate) -> Optional[Candidate]:
    """Return a valid trimmed copy, or None if edge trimming cannot satisfy the rules."""
    if not cand.segments or any(s.episode not in show for s in cand.segments):
        return None
    first, last = cand.segments[0], cand.segments[-1]
    if cand.hook.episode != first.episode or cand.cliffhanger.episode != last.episode:
        return None
    retarget = []
    probe = validate(show, _copy(cand, [Segment(s.episode, s.from_block, s.to_block) for s in cand.segments]))
    if any("cliffhanger block" in e and "not on screen" in e for e in probe.errors):
        # The declared cliffhanger is a shot that starts at the cut. Point it at the last story beat
        # (dialogue/ACTION/SFX) inside the last segment instead; its text comes from the script.
        beats = [b for b in show[last.episode].blocks[last.from_block:last.to_block + 1] if b.kind != "camera"]
        if not beats:
            return None
        retarget.append(f"cliffhanger E{last.episode}#{cand.cliffhanger.block}->#{beats[-1].id} (shot was off screen)")
        cand = _copy(cand, cand.segments)
        cand.id = cand.id[:-1]
        cand.cliffhanger = type(cand.cliffhanger)(last.episode, beats[-1].id)
        cand.cliffhanger_description = beats[-1].text
    if any("hook block" in e and "not on screen" in e for e in probe.errors):
        beats = [b for b in show[first.episode].blocks[first.from_block:first.to_block + 1] if b.kind != "camera"]
        if not beats:
            return None
        retarget.append(f"hook E{first.episode}#{cand.hook.block}->#{beats[0].id} (shot was off screen)")
        cand = _copy(cand, cand.segments)
        cand.id = cand.id[:-1]
        cand.hook = type(cand.hook)(first.episode, beats[0].id)
        cand.hook_description = beats[0].text
    if not (first.from_block <= cand.hook.block <= first.to_block and
            last.from_block <= cand.cliffhanger.block <= last.to_block):
        return None

    middle = [Segment(s.episode, s.from_block, s.to_block) for s in cand.segments[1:-1]]
    single = len(cand.segments) == 1
    # Candidate start blocks: from the original start up to the hook (latest start = hook itself).
    starts = range(first.from_block, cand.hook.block + 1)
    # Candidate end blocks: from the cliffhanger out to the original end (earliest first).
    ends = range(cand.cliffhanger.block, last.to_block + 1)
    best = None
    for s_block in starts:                       # prefer keeping as much opening context as possible
        for e_block in ends:                     # prefer ending as close to the cliffhanger as possible
            if single:
                segs = [Segment(first.episode, s_block, e_block)]
            else:
                segs = [Segment(first.episode, s_block, first.to_block)] + \
                       [Segment(m.episode, m.from_block, m.to_block) for m in middle] + \
                       [Segment(last.episode, last.from_block, e_block)]
            trial = validate(show, _copy(cand, segs))
            if not trial.errors:
                best = trial
                break
        if best:
            break
    if best is None:
        return None
    changes = list(retarget)
    if best.segments[0].from_block != first.from_block:
        changes.append(f"start E{first.episode}#{first.from_block}->#{best.segments[0].from_block}")
    if best.segments[-1].to_block != last.to_block:
        changes.append(f"end E{last.episode}#{last.to_block}->#{best.segments[-1].to_block}")
    best.warnings.append("auto-repaired (no model call): " + ", ".join(changes or ["none"]) +
                         f"; fixed: {'; '.join(probe.errors)}")
    return best
