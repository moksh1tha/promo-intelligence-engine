"""Promo candidate data model and deterministic validation.

A candidate is a list of segments, each a contiguous range of block ids in
one episode. Validation resolves block ids to exact source timestamps and
checks every rule that can be checked without judgement:

  * episodes / block ids exist
  * segments are non-overlapping and chronological within each episode
  * total runtime is 20-60 s
  * the declared hook starts within the first 3 s of the promo
  * the declared cliffhanger is in the last segment and the clip ends
    within a few seconds of it (no trailing resolution)

Judgement calls (is the hook actually strong? is context missing?) are left
to the critic stage.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .screenplay import Block, Episode

MIN_DURATION = 20.0
MAX_DURATION = 60.0
HOOK_WINDOW = 3.0
CLIFF_TAIL = 4.0


@dataclass
class Segment:
    episode: int
    from_block: int
    to_block: int
    start: float = 0.0
    end: float = 0.0

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


@dataclass
class BlockRef:
    episode: int
    block: int


@dataclass
class Candidate:
    id: str
    origin: str                      # e.g. "episode-3", "arc-2", "heuristic-ep3"
    segments: List[Segment]
    hook: BlockRef
    hook_description: str
    cliffhanger: BlockRef
    cliffhanger_description: str
    why_it_works: str
    context_note: str = ""
    self_scores: Dict[str, float] = field(default_factory=dict)
    # filled by validation
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    hook_offset: Optional[float] = None
    cliff_tail: Optional[float] = None
    # filled by critic / ranking
    critique: Dict = field(default_factory=dict)
    scores: Dict[str, int] = field(default_factory=dict)
    total_score: int = 0
    rejection_reasons: List[str] = field(default_factory=list)
    borderline_reasons: List[str] = field(default_factory=list)
    repaired_from: Optional[str] = None
    duplicate_of: Optional[str] = None
    public_id: str = ""
    review: Dict = field(default_factory=dict)

    @property
    def total_duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)

    @property
    def episodes(self) -> List[int]:
        out = []
        for s in self.segments:
            if s.episode not in out:
                out.append(s.episode)
        return out

    def intervals(self) -> List[Tuple[int, float, float]]:
        return [(s.episode, s.start, s.end) for s in self.segments]


def segment_end(blocks: List[Block]) -> float:
    """A segment ends when its last story beat (dialogue/action/SFX) ends.

    CAMERA blocks describe shots; a shot may be trimmed by the cut, so a long
    camera block does not stretch the segment. Dialogue/action/SFX are never
    cut mid-way: the segment runs until the latest of them has finished.
    """
    story = [b.end for b in blocks if b.kind != "camera"]
    return max(story) if story else max(b.end for b in blocks)


def _segment_blocks(show: Dict[int, Episode], seg: Segment) -> List[Block]:
    return show[seg.episode].blocks[seg.from_block: seg.to_block + 1]


MIN_VISIBLE = 0.3   # a block must be on screen at least this long to count as part of the clip


def timeline(show: Dict[int, Episode], cand: Candidate):
    """(relative_start, relative_end, segment_index, block) for every block actually on screen, in viewing order.

    Times are clip-relative and clipped to the segment: a CAMERA shot that runs past the cut is shown
    only up to the cut, and a block that would be on screen for less than MIN_VISIBLE seconds (e.g. a
    camera shot starting exactly at the cut) is not part of the clip at all.
    """
    offset = 0.0
    out = []
    for i, seg in enumerate(cand.segments):
        for b in _segment_blocks(show, seg):
            visible = min(b.end, seg.end) - b.start
            if visible < MIN_VISIBLE:
                continue
            out.append((round(offset + b.start - seg.start, 2), round(offset + min(b.end, seg.end) - seg.start, 2), i, b))
        offset += seg.duration
    return out


def render_clip(show: Dict[int, Episode], cand: Candidate, include_camera: bool = True) -> str:
    """What a first-time viewer sees, with clip-relative times."""
    lines = []
    prev_seg = None
    for rs, re_, i, b in timeline(show, cand):
        if i != prev_seg:
            seg = cand.segments[i]
            lines.append(f"--- CUT {i + 1}: episode {seg.episode} (source {_fmt(seg.start)}-{_fmt(seg.end)}) ---")
            prev_seg = i
        if b.kind == "camera" and not include_camera:
            continue
        if b.kind == "dialogue":
            who = b.speaker + (f" ({b.tone})" if b.tone else "")
        else:
            who = b.kind.upper()
        lines.append(f"[{_fmt(rs)}-{_fmt(re_)}] {who}: {b.text}")
    return "\n".join(lines)


def _fmt(sec: float) -> str:
    from .timecode import fmt_short
    return fmt_short(sec)


def validate(show: Dict[int, Episode], cand: Candidate) -> Candidate:
    """Resolve timestamps and record deterministic rule violations on the candidate."""
    cand.errors, cand.warnings = [], []
    if not cand.segments:
        cand.errors.append("no segments")
        return cand

    for i, seg in enumerate(cand.segments, 1):
        ep = show.get(seg.episode)
        if ep is None:
            cand.errors.append(f"segment {i}: episode {seg.episode} does not exist")
            continue
        n = len(ep.blocks)
        if not (0 <= seg.from_block < n and 0 <= seg.to_block < n):
            cand.errors.append(f"segment {i}: block range #{seg.from_block}-#{seg.to_block} outside episode {seg.episode} (0-{n - 1})")
            continue
        if seg.to_block < seg.from_block:
            cand.errors.append(f"segment {i}: to_block #{seg.to_block} is before from_block #{seg.from_block}")
            continue
        blocks = _segment_blocks(show, seg)
        seg.start = blocks[0].start
        seg.end = segment_end(blocks)
    if cand.errors:
        return cand

    # never cut a line of dialogue mid-way (a line outside the range can still be running at a cut)
    for i, seg in enumerate(cand.segments, 1):
        for b in show[seg.episode].blocks:
            if b.kind != "dialogue":
                continue
            if b.start < seg.start < b.end:
                cand.errors.append(f"segment {i} starts mid-line (E{seg.episode}#{b.id} runs {b.start:.2f}-{b.end:.2f})")
            if b.start < seg.end < b.end:
                cand.errors.append(f"segment {i} ends mid-line (E{seg.episode}#{b.id} runs {b.start:.2f}-{b.end:.2f})")

    # chronological + non-overlapping within each episode
    last_end: Dict[int, float] = {}
    for i, seg in enumerate(cand.segments, 1):
        if seg.episode in last_end and seg.start < last_end[seg.episode]:
            cand.errors.append(f"segment {i}: episode {seg.episode} segments overlap or are out of chronological order")
        last_end[seg.episode] = seg.end
    # episodes should appear in story order unless the model deliberately flash-forwards
    eps = [s.episode for s in cand.segments]
    if eps != sorted(eps):
        cand.warnings.append("segments jump backwards in story order (possible misleading edit)")

    total = cand.total_duration
    if total < MIN_DURATION:
        cand.errors.append(f"duration {total:.2f}s is below {MIN_DURATION:.0f}s")
    if total > MAX_DURATION:
        cand.errors.append(f"duration {total:.2f}s exceeds {MAX_DURATION:.0f}s")

    tl = timeline(show, cand)
    clip_end = total

    hook = [t for t in tl if t[3].id == cand.hook.block and cand.segments[t[2]].episode == cand.hook.episode]
    if not hook:
        cand.errors.append(f"hook block E{cand.hook.episode}#{cand.hook.block} is not on screen in the clip")
    else:
        cand.hook_offset = hook[0][0]
        if cand.hook_offset > HOOK_WINDOW:
            cand.errors.append(f"hook starts at {cand.hook_offset:.2f}s, outside the first {HOOK_WINDOW:.0f}s")

    cliff = [t for t in tl if t[3].id == cand.cliffhanger.block and cand.segments[t[2]].episode == cand.cliffhanger.episode]
    if not cliff:
        cand.errors.append(f"cliffhanger block E{cand.cliffhanger.episode}#{cand.cliffhanger.block} is not on screen "
                           f"in the clip (it starts at or after the cut)")
    else:
        rs, re_, seg_i, _ = cliff[-1]
        cand.cliff_tail = round(clip_end - re_, 2)
        if seg_i != len(cand.segments) - 1:
            cand.errors.append("cliffhanger is not in the final segment")
        elif cand.cliff_tail > CLIFF_TAIL:
            cand.errors.append(f"clip runs {cand.cliff_tail:.2f}s past the cliffhanger (max {CLIFF_TAIL:.0f}s)")
    return cand


def duration_quality(total: float) -> int:
    """0-5 points for duration / boundary quality (deterministic)."""
    if total < MIN_DURATION or total > MAX_DURATION:
        return 0
    if 25 <= total <= 50:
        return 5
    if total <= 55:
        return 4
    return 3


def overlap_seconds(a: Candidate, b: Candidate) -> float:
    total = 0.0
    for ea, sa, xa in a.intervals():
        for eb, sb, xb in b.intervals():
            if ea == eb:
                total += max(0.0, min(xa, xb) - max(sa, sb))
    return total
