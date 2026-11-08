"""Human-review flags for promos. Runs locally, no model calls.

Every flag says what kind of check produced it, so objective facts are never
mixed up with judgement calls:

  objective    - computed exactly from the screenplay blocks (timing, visibility, order, scores present)
  approximate  - text matching between a model's description and the clip's blocks; a heuristic
                 that can misfire on paraphrases, so it can only ask for a look, never reject
  subjective   - comes from the LLM critic's judgement (flags it raised, scout/critic disagreement)

Each flag carries a suggested action: accept, inspect, repair (metadata can be corrected from the
source) or reject. A promo's review status is the most serious action among its flags.
"""
import re
from typing import Dict, List

from .candidates import BlockRef, Candidate, HOOK_WINDOW, timeline
from .critic import SCORE_CAPS
from .screenplay import Episode

DESC_SUPPORT_MIN = 0.25     # share of a description's content words found in the relevant blocks
DISAGREEMENT = 15           # scout self-score vs critic score, points out of 95
LOW_COHERENCE = 3           # out of 10
CLIFF_WINDOW = 6.0          # seconds at the end of the clip searched for the cliffhanger description

ACTION_ORDER = {"accept": 0, "inspect": 1, "repair": 2, "reject": 3}
STATUS = {"accept": "recommended", "inspect": "needs_review", "repair": "fix_metadata", "reject": "not_recommended"}

_STOP = set("""the a an and or but of to in on at for with as by from into onto over under her his their its she he
they them this that these those is are was were be been being has have had will would can could while then than so
very just not no yet who whom what when where which how why all any some more most less other such own same both each
few out up down off again once here there about after before between through during only also too it""".split())


def _words(text: str) -> set:
    out = set()
    for w in re.findall(r"[a-z']+", text.lower()):
        w = w.strip("'")
        if len(w) >= 4 and w not in _STOP:
            out.add(re.sub(r"(ing|ed|es|s|ly)$", "", w))
    return out


def description_support(description: str, texts: List[str]) -> float:
    """Approximate: fraction of the description's content words that appear in the given block texts."""
    d = _words(description)
    if not d:
        return 1.0
    return len(d & _words(" ".join(texts))) / len(d)


def _flag(code, kind, action, detail, blocks=None):
    return {"code": code, "kind": kind, "action": action, "detail": detail, "blocks": blocks or []}


def _block_line(show, ep: int, t) -> str:
    rs, re_, _, b = t
    who = b.speaker if b.kind == "dialogue" else b.kind.upper()
    return f"{rs:.1f}-{re_:.1f}s  E{ep}#{b.id} {who}: {b.text[:110]}"


def offscreen_blocks(show: Dict[int, Episode], cand: Candidate) -> List[str]:
    """Blocks inside a segment's id range that are never on screen (they start at or after the cut)."""
    on_screen = {(cand.segments[t[2]].episode, t[3].id) for t in timeline(show, cand)}
    out = []
    for seg in cand.segments:
        for b in show[seg.episode].blocks[seg.from_block: seg.to_block + 1]:
            if (seg.episode, b.id) not in on_screen:
                out.append(f"E{seg.episode}#{b.id} {b.kind.upper()} starts {b.start - seg.end:+.1f}s after the cut: {b.text[:100]}")
    return out


def review(show: Dict[int, Episode], cand: Candidate, critic_saw_offscreen: bool = False) -> Dict:
    """Return {"status", "action", "flags", "proposed_fixes"} for a validated candidate.

    critic_saw_offscreen: set when auditing runs made before v0.2, whose critic prompt also listed
    blocks that start at or after the cut; their scores may credit content the viewer never sees.
    """
    flags: List[dict] = []
    fixes: Dict[str, dict] = {}
    tl = timeline(show, cand)
    total = cand.total_duration
    ep_of = lambda t: cand.segments[t[2]].episode  # noqa: E731

    # ---- objective: hook -------------------------------------------------------
    hook_t = [t for t in tl if t[3].id == cand.hook.block and ep_of(t) == cand.hook.episode]
    story = [t for t in tl if t[3].kind != "camera"]
    if not hook_t:
        early = [t for t in story if t[0] <= HOOK_WINDOW]
        if early:
            best = early[0]
            fixes["hook"] = {"episode": ep_of(best), "block": best[3].id, "description": best[3].text}
            flags.append(_flag("hook_not_on_screen", "objective", "repair",
                               f"declared hook E{cand.hook.episode}#{cand.hook.block} is not on screen; first story beat "
                               f"inside 3 s is E{ep_of(best)}#{best[3].id}", [_block_line(show, ep_of(best), best)]))
        else:
            flags.append(_flag("hook_absent", "objective", "reject",
                               "declared hook is not on screen and no dialogue/action/SFX starts in the first 3 s"))
    else:
        h = hook_t[0]
        if h[0] > HOOK_WINDOW:
            flags.append(_flag("hook_late", "objective", "reject",
                               f"hook starts at {h[0]:.1f}s (limit {HOOK_WINDOW:.0f}s)", [_block_line(show, ep_of(h), h)]))
        first_story = story[0][0] if story else None
        if h[3].kind == "camera" and (first_story is None or first_story > HOOK_WINDOW):
            flags.append(_flag("hook_visual_only", "objective", "inspect",
                               f"hook is a CAMERA shot description; first line/action/SFX starts at "
                               f"{first_story:.1f}s. Confirm in the rendered video that the opening shot hooks on its own.",
                               [_block_line(show, ep_of(h), h)] + ([_block_line(show, ep_of(story[0]), story[0])] if story else [])))
        sup = description_support(cand.hook_description, [t[3].text for t in tl if t[0] <= HOOK_WINDOW])
        if cand.hook_description and sup < DESC_SUPPORT_MIN:
            flags.append(_flag("hook_description_unsupported", "approximate", "inspect",
                               f"only {sup:.0%} of the hook description's key words appear in the first 3 s "
                               f"(text match, may be a paraphrase): \"{cand.hook_description[:120]}\"",
                               [_block_line(show, ep_of(t), t) for t in tl if t[0] <= HOOK_WINDOW]))

    # ---- objective: cliffhanger ------------------------------------------------
    cliff_t = [t for t in tl if t[3].id == cand.cliffhanger.block and ep_of(t) == cand.cliffhanger.episode]
    last_story = story[-1] if story else None
    if not cliff_t:
        if last_story is not None:
            fixes["cliffhanger"] = {"episode": ep_of(last_story), "block": last_story[3].id,
                                    "description": last_story[3].text}
        flags.append(_flag("cliffhanger_not_on_screen", "objective", "repair",
                           f"declared cliffhanger E{cand.cliffhanger.episode}#{cand.cliffhanger.block} starts at or after "
                           f"the cut, so it is never shown. The clip actually ends on: "
                           f"{last_story[3].text[:120] if last_story else '(nothing)'}",
                           [_block_line(show, ep_of(t), t) for t in tl if t[1] >= total - CLIFF_WINDOW]))
    sup = description_support(cand.cliffhanger_description, [t[3].text for t in tl if t[1] >= total - CLIFF_WINDOW])
    if cand.cliffhanger_description and sup < DESC_SUPPORT_MIN:
        flags.append(_flag("cliffhanger_description_unsupported", "approximate", "inspect",
                           f"only {sup:.0%} of the cliffhanger description's key words appear in the last "
                           f"{CLIFF_WINDOW:.0f} s (text match, may be a paraphrase): \"{cand.cliffhanger_description[:120]}\"",
                           [_block_line(show, ep_of(t), t) for t in tl if t[1] >= total - CLIFF_WINDOW]))

    # ---- objective: split lines at the edges ------------------------------------
    if story and story[0][3].kind == "dialogue" and re.match(r"^\W*(\.\.\.|…)", story[0][3].text):
        flags.append(_flag("opens_mid_sentence", "objective", "inspect",
                           "first line is the second half of a sentence split across blocks; consider starting one block earlier",
                           [_block_line(show, ep_of(story[0]), story[0])]))
    if story and story[-1][3].kind == "dialogue" and re.search(r"(\.\.\.|…)\W*$", story[-1][3].text):
        flags.append(_flag("ends_mid_sentence", "objective", "inspect",
                           "last line trails off into a sentence continued in the next block (may be deliberate)",
                           [_block_line(show, ep_of(story[-1]), story[-1])]))

    # ---- objective: order and timestamp consistency -------------------------------
    eps = [s.episode for s in cand.segments]
    if eps != sorted(eps):
        flags.append(_flag("story_order", "objective", "reject", f"segments run out of story order: episodes {eps}"))
    if len(set(eps)) > 1:
        flags.append(_flag("cross_episode", "objective", "inspect",
                           f"combines episodes {sorted(set(eps))}; check the cuts read as one story to a new viewer"))
    shown = round(sum(s.duration for s in cand.segments), 2)
    tl_end = max((t[1] for t in tl), default=0.0)
    if tl and tl_end - shown > 0.011:
        flags.append(_flag("timestamp_mismatch", "objective", "reject",
                           f"displayed clip runs to {tl_end:.2f}s but the segments total {shown:.2f}s"))

    # ---- critic scores ---------------------------------------------------------
    if critic_saw_offscreen:
        off = offscreen_blocks(show, cand)
        if off:
            flags.append(_flag("critic_saw_offscreen_blocks", "objective", "inspect",
                               "the critic's prompt included shot(s) that are never on screen, so its scores "
                               "(especially the cliffhanger) may credit content the viewer does not see", off[:3]))
    crit = cand.critique or {}
    missing = crit.get("missing_scores") or []
    if missing:
        flags.append(_flag("critic_score_missing", "objective", "inspect",
                           f"critic reply had no readable score for {', '.join(missing)} (counted as 0 in the total)"))
    coh = cand.scores.get("narrative_coherence")
    if coh is not None and "narrative_coherence" not in missing:
        if coh == 0:
            flags.append(_flag("zero_coherence", "objective", "inspect",
                               "critic scored narrative coherence 0/10; do not publish without a human check"))
        elif coh <= LOW_COHERENCE:
            flags.append(_flag("low_coherence", "subjective", "inspect", f"critic scored narrative coherence {coh}/10"))
    for r in cand.borderline_reasons:
        flags.append(_flag("borderline_subjective", "subjective", "inspect",
                           f"kept for human review instead of rejected: {r} (otherwise scores well, "
                           f"total {cand.total_score})"))
    for w in crit.get("warnings") or []:
        code = "misleading_edit" if "misleading" in w else "missing_context" if "context" in w else "critic_warning"
        flags.append(_flag(code, "subjective", "inspect", w))
    if cand.self_scores and cand.scores:
        dims = [k for k in SCORE_CAPS if k in cand.self_scores]
        if dims:
            self_total = sum(cand.self_scores[k] for k in dims)
            crit_total = sum(cand.scores[k] for k in dims)
            if abs(self_total - crit_total) >= DISAGREEMENT:
                flags.append(_flag("scout_critic_disagreement", "subjective", "inspect",
                                   f"scout rated {self_total:.0f}, blind critic {crit_total} on the same "
                                   f"{len(dims)} dimensions"))

    action = max((f["action"] for f in flags), key=ACTION_ORDER.get, default="accept")
    return {"status": STATUS[action], "action": action, "flags": flags, "proposed_fixes": fixes}


def apply_fixes(cand: Candidate, fixes: Dict[str, dict]) -> List[str]:
    """Apply metadata corrections that the source supports. Returns a changelog. Never changes the cut."""
    log = []
    if "hook" in fixes:
        f = fixes["hook"]
        log.append(f"hook E{cand.hook.episode}#{cand.hook.block} -> E{f['episode']}#{f['block']} "
                   f"(old description: {cand.hook_description!r})")
        cand.hook = BlockRef(f["episode"], f["block"])
        cand.hook_description = f["description"]
    if "cliffhanger" in fixes:
        f = fixes["cliffhanger"]
        log.append(f"cliffhanger E{cand.cliffhanger.episode}#{cand.cliffhanger.block} -> E{f['episode']}#{f['block']} "
                   f"(old description: {cand.cliffhanger_description!r})")
        cand.cliffhanger = BlockRef(f["episode"], f["block"])
        cand.cliffhanger_description = f["description"]
    return log
