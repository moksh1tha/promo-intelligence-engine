"""Clip-level reuse of saved critic reviews. No model calls.

The response cache is keyed by the whole prompt, and a batch prompt holds 15 clips, so changing
one clip in a batch would re-review all 15. This memo reads every cached batch-critic exchange,
splits it back into clips, identifies exactly which screenplay blocks each clip showed the critic
(in order), and maps that block sequence -> review.

A new clip reuses a review only if it shows the critic exactly the same blocks in the same order.
Display details such as a trimmed shot's end time do not matter, but a clip that previously
included a shot that is never on screen (fixed in v0.2) shows a different block sequence now, so
it is reviewed again instead of inheriting a score that credited unseen content.
"""
import glob
import json
import os
import re
from typing import Dict, Optional, Tuple

from .candidates import Candidate, timeline
from .llm import extract_json
from .screenplay import Episode

CLIP_RE = re.compile(r"=== CLIP (C\d+) \([\d.]+ s\) ===\n(.*?)(?=\n\n=== CLIP |\n\nFor EACH clip)", re.S)
CUT_RE = re.compile(r"^--- CUT \d+: episode (\d+) ")
LINE_RE = re.compile(r"^\[[\d:.]+-[\d:.]+\] (.*)$")


def clip_block_key(show: Dict[int, Episode], cand: Candidate) -> Tuple:
    return tuple((cand.segments[t[2]].episode, t[3].id) for t in timeline(show, cand))


def _parse_clip_text(show: Dict[int, Episode], text: str) -> Optional[Tuple]:
    key, ep, cursor = [], None, 0
    for line in text.splitlines():
        m = CUT_RE.match(line)
        if m:
            ep, cursor = int(m.group(1)), 0
            if ep not in show:
                return None
            continue
        m = LINE_RE.match(line)
        if not m or ep is None:
            continue
        body = m.group(1)
        blocks = show[ep].blocks
        hit = next((b for b in blocks[cursor:] if body.endswith(": " + b.text)), None)
        if hit is None:
            return None
        key.append((ep, hit.id))
        cursor = hit.id + 1
    return tuple(key) or None


class CritiqueMemo:
    def __init__(self, cache_dir: Optional[str], provider: str, model: str, effort: Optional[str],
                 show: Optional[Dict[int, Episode]] = None):
        self.reviews: Dict[Tuple, dict] = {}
        self.show = show
        if not cache_dir or not os.path.isdir(cache_dir) or show is None:
            return
        for f in glob.glob(os.path.join(cache_dir, "*.json")):
            try:
                with open(f, encoding="utf-8") as fh:
                    d = json.load(fh)
            except (OSError, ValueError):
                continue
            if not isinstance(d, dict) or (d.get("provider"), d.get("model"), d.get("effort")) != (provider, model, effort):
                continue
            prompt = d.get("prompt", "")
            if "=== CLIP " not in prompt:
                continue
            try:
                payload = extract_json(d.get("response", ""))
            except Exception:
                continue
            by_id = {str(r.get("id")): r for r in (payload.get("reviews", []) if isinstance(payload, dict) else [])}
            for cid, text in CLIP_RE.findall(prompt):
                key = _parse_clip_text(show, text)
                if key and cid in by_id:
                    self.reviews[key] = by_id[cid]

    def get(self, cand: Candidate) -> Optional[dict]:
        if self.show is None:
            return None
        r = self.reviews.get(clip_block_key(self.show, cand))
        return dict(r) if r is not None else None
