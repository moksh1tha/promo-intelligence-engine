"""Parse timestamped microdrama screenplays into structured episodes.

Every block in the source (CAMERA / ACTION / SOUND FX / dialogue) gets a
stable integer id within its episode, ordered by start time. Models select
clips by block id, never by typing timestamps, so every timestamp in the
output is copied from the source file.
"""
import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import timecode

_BLOCK_RE = re.compile(
    r"^\*\*(?P<label>.+?):\*\*\s*\n"
    r"Start: \((?P<start>[\d:.]+)\) \| End: \((?P<end>[\d:.]+)\)\s*\n"
    r"(?P<body>.*?)(?=\n\s*\n|\n\*\*|\n---|\Z)",
    re.S | re.M,
)
_HEADER_RE = {
    "number": re.compile(r"^EPISODE NUMBER:\s*(.*)$", re.M),
    "logline": re.compile(r"^EPISODE LOGLINE:\s*(.*)$", re.M),
    "continuity": re.compile(r"^CONTINUITY NOTES:\s*(.*)$", re.M),
}
_SCENE_RE = re.compile(
    r"^SCENE (?P<num>\d+)\s*\nScene heading:\s*(?P<heading>.*)\nScene start:\s*(?P<start>\S+)\s*\nScene end:\s*(?P<end>\S+)",
    re.M,
)


@dataclass
class Block:
    id: int
    kind: str          # camera | action | sfx | dialogue
    label: str         # raw label, e.g. "LADY JULIANA (Confused, Breathless)"
    speaker: str       # for dialogue: "LADY JULIANA"; otherwise the kind in caps
    tone: str          # parenthetical, may be ""
    start: float
    end: float
    text: str
    scene: int = 0

    def render(self, offset: float = 0.0) -> str:
        span = f"{timecode.fmt_short(self.start - offset)}-{timecode.fmt_short(self.end - offset)}"
        if self.kind == "dialogue":
            who = self.speaker + (f" ({self.tone})" if self.tone else "")
            return f"#{self.id} [{span}] {who}: {self.text}"
        return f"#{self.id} [{span}] {self.kind.upper()}: {self.text}"


@dataclass
class Scene:
    number: int
    heading: str
    start: float
    end: Optional[float]


@dataclass
class Episode:
    number: int
    source_file: str
    logline: str
    continuity: str
    scenes: List[Scene]
    blocks: List[Block]
    warnings: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        ends = [b.end for b in self.blocks] + [s.end for s in self.scenes if s.end is not None]
        return max(ends) if ends else 0.0

    def block(self, block_id: int) -> Block:
        return self.blocks[block_id]

    def render(self, include_camera: bool = True) -> str:
        lines = [
            f"EPISODE {self.number}  (runtime {timecode.fmt_short(self.duration)})",
            f"LOGLINE: {self.logline}",
            "",
        ]
        current_scene = None
        for b in self.blocks:
            if b.scene != current_scene:
                current_scene = b.scene
                sc = next((s for s in self.scenes if s.number == b.scene), None)
                if sc:
                    lines.append(f"-- SCENE {sc.number}: {sc.heading}")
            if b.kind == "camera" and not include_camera:
                continue
            lines.append(b.render())
        return "\n".join(lines)


def _classify(label: str):
    m = re.match(r"^(?P<who>[^(]+?)\s*(?:\((?P<tone>.*)\))?\s*$", label.strip())
    who = (m.group("who") if m else label).strip()
    tone = ((m.group("tone") if m else "") or "").strip()
    upper = who.upper()
    if upper == "CAMERA":
        return "camera", "CAMERA", tone
    if upper == "ACTION":
        return "action", "ACTION", tone
    if upper in ("SOUND FX", "SFX", "SOUND"):
        return "sfx", "SFX", tone
    return "dialogue", who, tone


def parse_episode(path: str) -> Episode:
    text = open(path, encoding="utf-8").read()
    warnings = []
    num_match = re.search(r"episode-(\d+)", os.path.basename(path))
    header_num = _HEADER_RE["number"].search(text)
    number = int(num_match.group(1)) if num_match else None
    if header_num:
        digits = re.findall(r"\d+", header_num.group(1))
        if digits and number is not None and int(digits[0]) != number:
            warnings.append(f"file name says episode {number}, header says {header_num.group(1)}")
        if number is None and digits:
            number = int(digits[0])
    if number is None:
        raise ValueError(f"cannot determine episode number for {path}")

    scenes = []
    for m in _SCENE_RE.finditer(text):
        try:
            end = timecode.parse(m.group("end"))
        except ValueError:
            end = None
            warnings.append(f"scene {m.group('num')} has unparseable end '{m.group('end')}'")
        scenes.append(Scene(int(m.group("num")), m.group("heading").strip(),
                            timecode.parse(m.group("start")), end))

    # Assign each block to the scene whose header precedes it in the file.
    scene_positions = [(m.start(), int(m.group("num"))) for m in _SCENE_RE.finditer(text)]

    raw = []
    for m in _BLOCK_RE.finditer(text):
        kind, speaker, tone = _classify(m.group("label"))
        start, end = timecode.parse(m.group("start")), timecode.parse(m.group("end"))
        if end < start:
            warnings.append(f"block '{m.group('label')}' at {m.group('start')} ends before it starts")
        scene_no = 0
        for pos, sn in scene_positions:
            if pos < m.start():
                scene_no = sn
        body = " ".join(m.group("body").split())
        raw.append((start, len(raw), kind, m.group("label").strip(), speaker, tone, end, body, scene_no))

    raw.sort(key=lambda r: (r[0], r[1]))
    blocks = [
        Block(id=i, kind=r[2], label=r[3], speaker=r[4], tone=r[5], start=r[0], end=r[6], text=r[7], scene=r[8])
        for i, r in enumerate(raw)
    ]
    if not blocks:
        warnings.append("no blocks parsed")

    logline = _HEADER_RE["logline"].search(text)
    continuity = _HEADER_RE["continuity"].search(text)
    return Episode(
        number=number,
        source_file=os.path.basename(path),
        logline=logline.group(1).strip() if logline else "",
        continuity=continuity.group(1).strip() if continuity else "",
        scenes=scenes,
        blocks=blocks,
        warnings=warnings,
    )


def load_show(data_dir: str) -> Dict[int, Episode]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"no .txt screenplays found in {data_dir}")
    episodes = {}
    for f in files:
        ep = parse_episode(f)
        if ep.number in episodes:
            raise ValueError(f"duplicate episode {ep.number}: {f} and {episodes[ep.number].source_file}")
        episodes[ep.number] = ep
    return dict(sorted(episodes.items()))
