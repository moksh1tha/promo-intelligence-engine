"""Build candidates from source timestamps (for reference picks and critic probes)."""
import json
from typing import Dict, List, Optional, Tuple

from . import timecode
from .candidates import BlockRef, Candidate, Segment, segment_end, validate
from .screenplay import Episode


def segment_from_times(show: Dict[int, Episode], episode: int, start: str, end: str) -> Segment:
    ep = show[episode]
    s, e = timecode.parse(start), timecode.parse(end)
    starts = [b.id for b in ep.blocks if b.start == s]
    if not starts:
        raise ValueError(f"E{episode}: no block starts at {start}")
    first = min(starts)
    best: Optional[int] = None
    for b in ep.blocks[first:]:
        if b.start >= e:
            break
        seg_end = segment_end(ep.blocks[first:b.id + 1])
        if seg_end == e:
            best = b.id
        if seg_end > e:
            break
    if best is None:
        raise ValueError(f"E{episode}: no block range from {start} ends at {end}")
    return Segment(episode, first, best)


def candidate_from_times(show, cid: str, segs: List[Tuple[int, str, str]], note: str = "",
                         origin: str = "reference") -> Candidate:
    segments = [segment_from_times(show, *s) for s in segs]
    first, last = segments[0], segments[-1]
    first_blocks = show[first.episode].blocks[first.from_block:first.to_block + 1]
    first_story = next((b.id for b in first_blocks if b.kind != "camera"), first.from_block)
    last_blocks = show[last.episode].blocks[last.from_block:last.to_block + 1]
    last_story = next((b.id for b in reversed(last_blocks) if b.kind != "camera"), last.to_block)
    cand = Candidate(id=cid, origin=origin, segments=segments,
                     hook=BlockRef(first.episode, first_story), hook_description="",
                     cliffhanger=BlockRef(last.episode, last_story), cliffhanger_description="",
                     why_it_works=note)
    return validate(show, cand)


def load_fixture_file(show, path: str, origin: str) -> List[Tuple[Candidate, dict]]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    missing = sorted({s["episode"] for item in items for s in item["segments"]} - set(show))
    if missing:
        flag = "--reference" if origin == "reference" else "--probes"
        raise ValueError(f"{path} is for another show: episode(s) {missing[:5]} are not in the data. "
                         f"For new data pass {flag} '' (or a file made for that show).")
    out = []
    for item in items:
        segs = [(s["episode"], s["start"], s["end"]) for s in item["segments"]]
        cand = candidate_from_times(show, item["id"], segs, item.get("note", ""), origin=origin)
        out.append((cand, item))
    return out
