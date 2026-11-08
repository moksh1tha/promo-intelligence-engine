"""Stage 2 - quality control.

The critic sees ONLY the clip (with clip-relative times), never the episode
or the series, so it judges like a first-time viewer. Its scores replace the
scout's self-scores, which removes self-grading bias. Mandatory-requirement
failures become explicit rejection reasons.
"""
import re
from typing import Callable, Dict

from . import prompts
from .candidates import Candidate, duration_quality, render_clip, timeline
from .llm import extract_json
from .screenplay import Episode

SCORE_CAPS = {"hook_strength": 25, "standalone_clarity": 20, "cliffhanger_strength": 25,
              "emotional_intensity": 15, "narrative_coherence": 10}

# Thresholds below which a mandatory requirement counts as failed even if the
# critic did not raise the flag. Tuned conservatively; see SOLUTION.md.
MIN_HOOK = 14
MIN_CLARITY = 11
MIN_CLIFF = 14

# A critic flag only rejects a clip when the score it is about is also weak (below
# FLAG_CONFIRM of that score's maximum). A flag on a clip that still scores well
# on that dimension is kept as a warning instead. Without this, any single flag
# rejected clips scoring ~80/100 (both known-good calibration probes failed).
FLAG_CONFIRM = 0.6
FLAG_SCORES = {
    "weak_hook": ["hook_strength"],
    "missing_context": ["standalone_clarity"],
    "predictable_ending": ["cliffhanger_strength"],
    "low_curiosity": ["cliffhanger_strength", "emotional_intensity"],
    "misleading_edit": ["narrative_coherence"],
}

FLAG_REASONS = {
    "weak_hook": "weak hook",
    "missing_context": "missing context",
    "predictable_ending": "predictable/resolved ending",
    "low_curiosity": "does not generate curiosity",
    "misleading_edit": "misleading edit",
}


class LLMCritic:
    def __init__(self, ask: Callable[[str, str, str], str]):
        self.ask = ask

    def critique(self, show: Dict[int, Episode], cand: Candidate) -> Dict:
        prompt = prompts.CRITIC_TEMPLATE.format(duration=cand.total_duration, clip_text=render_clip(show, cand))
        safe = re.sub(r"[^A-Za-z0-9]+", "_", cand.id)
        return extract_json(self.ask(f"critic_{safe}", prompts.CRITIC_SYSTEM, prompt))


class BatchLLMCritic:
    """Several clips per request. Clips get neutral ids (C1, C2 ...) so nothing about their
    origin leaks, and the prompt tells the model to judge each one independently."""

    def __init__(self, ask: Callable[[str, str, str], str], per_call: int = 15):
        self.ask = ask
        self.per_call = max(1, per_call)

    def critique_many(self, show: Dict[int, Episode], cands, stage: str = "critic_batch") -> Dict[str, Dict]:
        results: Dict[str, Dict] = {}
        for k in range(0, len(cands), self.per_call):
            chunk = cands[k:k + self.per_call]
            ids = {f"C{j + 1}": c for j, c in enumerate(chunk)}
            clips = "\n\n".join(f"=== CLIP {cid} ({c.total_duration:.1f} s) ===\n{render_clip(show, c)}"
                                 for cid, c in ids.items())
            prompt = prompts.CRITIC_BATCH_TEMPLATE.format(n=len(chunk), clips=clips)
            payload = extract_json(self.ask(f"{stage}{k // self.per_call + 1:02d}", prompts.CRITIC_SYSTEM, prompt))
            for review in (payload.get("reviews", []) if isinstance(payload, dict) else []):
                c = ids.get(str(review.get("id", "")).strip())
                if c is not None:
                    results[c.id] = review
        return results


class HeuristicCritic:
    """Model-free baseline critic using surface features. Coarse by design."""

    def critique(self, show, cand: Candidate) -> Dict:
        from .scout import intensity
        tl = timeline(show, cand)
        story = [t for t in tl if t[3].kind != "camera"]
        first = [t for t in story if t[0] <= 3.0]
        hook_i = max((intensity(t[3]) for t in first), default=0)
        speakers = {t[3].speaker for t in story if t[3].kind == "dialogue"}
        named = sum(1 for t in story if t[3].kind == "dialogue" and re.search(r"\b[A-Z][a-z]{2,}\b", t[3].text))
        last = story[-1][3] if story else None
        last_ep = show[cand.segments[-1].episode]
        story_ids = [b.id for b in last_ep.blocks if b.kind != "camera"]
        ends_episode = bool(last) and story_ids and last.id == story_ids[-1]
        cliff = 8 + (6 if last is not None and "?" in last.text else 0) + (6 if ends_episode else 0) \
            + min(5, intensity(last) if last else 0)
        mean_int = sum(intensity(t[3]) for t in story) / max(1, len(story))
        scores = {
            "hook_strength": min(25, 6 + 3 * hook_i),
            "standalone_clarity": min(20, 6 + 2 * min(len(speakers), 3) + min(named, 4)),
            "cliffhanger_strength": min(25, cliff),
            "emotional_intensity": min(15, int(4 + 3 * mean_int)),
            "narrative_coherence": 9 if len(cand.segments) == 1 else 6,
        }
        return {
            "scores": scores,
            "flags": {"weak_hook": hook_i < 2, "missing_context": len(speakers) < 2,
                      "predictable_ending": scores["cliffhanger_strength"] < MIN_CLIFF,
                      "low_curiosity": False, "misleading_edit": False},
            "what_viewer_understands": "(heuristic critic - no comprehension model)",
            "confusions": [],
            "open_question": last.text[:120] if last else "",
            "hook_assessment": f"opening intensity {hook_i}",
            "cliffhanger_assessment": "ends on the episode's final beat" if ends_episode else "mid-episode ending",
            "why_it_works": cand.why_it_works,
        }


# Borderline policy (default since v0.3). A single weak subjective dimension no longer rejects a
# clip that is otherwise good: it is kept and sent to human review. Clear failures still reject.
BORDERLINE_MIN_TOTAL = 75     # a one-weakness clip must score as well as the band of clips the strict rule accepts
REJECT_BELOW_TOTAL = 65       # any clip under this total is rejected
VERY_WEAK = {"hook_strength": 8, "standalone_clarity": 6, "cliffhanger_strength": 8}
_REASON_DIM = {"critic: weak hook": "hook", "hook score": "hook",
               "critic: missing context": "context", "clarity score": "context",
               "critic: predictable/resolved ending": "ending", "cliffhanger score": "ending",
               "critic: does not generate curiosity": "ending", "critic: misleading edit": "coherence"}


def _dim(reason: str) -> str:
    return next((d for k, d in _REASON_DIM.items() if reason.startswith(k)), reason)


def apply_critique(cand: Candidate, critique: Dict, policy: str = "borderline") -> Candidate:
    """policy="strict": any weak dimension rejects (v0.1/v0.2 behaviour).
    policy="borderline": clear failures reject; one weak dimension on an otherwise good clip
    (total >= BORDERLINE_MIN_TOTAL) keeps the clip with a human-review reason instead."""
    """Clamp scores, add deterministic duration score, decide pass/fail."""
    cand.critique = critique
    # Normalise keys: models occasionally emit e.g. "narrative_ coherence". A score that is
    # missing or unreadable is recorded (and flagged for human review), never silently
    # treated as a real 0 that passes or fails the clip.
    raw = {re.sub(r"[^a-z]", "", str(k).lower()): v for k, v in (critique.get("scores", {}) or {}).items()}
    scores, missing = {}, []
    for k, cap in SCORE_CAPS.items():
        try:
            v = int(round(float(raw[k.replace("_", "")])))
        except (KeyError, TypeError, ValueError):
            v, missing = 0, missing + [k]
        scores[k] = max(0, min(cap, v))
    critique["missing_scores"] = missing
    scores["duration_and_boundary_quality"] = duration_quality(cand.total_duration)
    cand.scores = scores
    cand.total_score = sum(scores.values())

    reasons, warnings, confirmed = [], [], []
    flags = critique.get("flags", {}) or {}
    for flag, label in FLAG_REASONS.items():
        if flags.get(flag) is not True:
            continue
        weak = [k for k in FLAG_SCORES[flag] if k not in missing and scores[k] < FLAG_CONFIRM * SCORE_CAPS[k]]
        if weak:
            confirmed.append(flag)
            reasons.append(f"critic: {label}")
        else:
            detail = ", ".join(f"{k} {scores[k]}/{SCORE_CAPS[k]}" for k in FLAG_SCORES[flag])
            warnings.append(f"critic flagged {label} but scored {detail}")
    critique["confirmed_flags"] = confirmed
    critique["warnings"] = warnings
    if scores["hook_strength"] < MIN_HOOK and "hook_strength" not in missing and "critic: weak hook" not in reasons:
        reasons.append(f"hook score {scores['hook_strength']}/25 below {MIN_HOOK}")
    if scores["standalone_clarity"] < MIN_CLARITY and "standalone_clarity" not in missing \
            and "critic: missing context" not in reasons:
        reasons.append(f"clarity score {scores['standalone_clarity']}/20 below {MIN_CLARITY}")
    if scores["cliffhanger_strength"] < MIN_CLIFF and "cliffhanger_strength" not in missing \
            and "critic: predictable/resolved ending" not in reasons:
        reasons.append(f"cliffhanger score {scores['cliffhanger_strength']}/25 below {MIN_CLIFF}")
    cand.borderline_reasons = []
    if policy == "strict" or not reasons:
        cand.rejection_reasons = reasons
        return cand
    dims = {_dim(r) for r in reasons}
    very_weak = [k for k, floor in VERY_WEAK.items() if k not in missing and scores[k] < floor]
    hard = []
    if cand.total_score < REJECT_BELOW_TOTAL:
        hard.append(f"total {cand.total_score} below {REJECT_BELOW_TOTAL}")
    if len(dims) >= 2:
        hard.append(f"weak on {len(dims)} dimensions ({', '.join(sorted(dims))})")
    if very_weak:
        hard.append("very weak " + ", ".join(f"{k} {scores[k]}" for k in very_weak))
    if not hard and cand.total_score < BORDERLINE_MIN_TOTAL:
        hard.append(f"one weak dimension and total {cand.total_score} below {BORDERLINE_MIN_TOTAL}")
    if hard:
        cand.rejection_reasons = reasons + hard
    else:
        cand.rejection_reasons = []
        cand.borderline_reasons = reasons
    return cand
