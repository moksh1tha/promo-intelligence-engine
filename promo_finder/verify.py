"""Independent post-hoc verification of a promos.json against the raw screenplay text.

Deliberately does NOT reuse the main parser: it re-reads the source files with
its own minimal regex so a parser bug cannot hide a timestamp error.

Checks per promo:
  * every segment start equals the start of a real block in that episode
  * every segment end equals the end of a real block in that episode
  * no dialogue line is cut mid-way at either boundary
  * duration_seconds matches end - start; total is 20-60 s
  * segments from one episode are chronological and non-overlapping
  * scoring adds up to total_score; ranks are 1..n in score order
"""
import glob
import json
import os
import re
import sys

BLOCK_RE = re.compile(r"\*\*(?P<label>[^*]+):\*\*\s*\nStart: \((?P<s>[\d:.]+)\) \| End: \((?P<e>[\d:.]+)\)")


def _sec(ts: str) -> float:
    total = 0.0
    for p in ts.split(":"):
        total = total * 60 + float(p)
    return round(total, 2)


def _load(data_dir):
    eps = {}
    for f in glob.glob(os.path.join(data_dir, "*.txt")):
        m = re.search(r"episode-(\d+)", os.path.basename(f))
        if not m:
            continue
        blocks = []
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        for b in BLOCK_RE.finditer(text):
            label = b.group("label").strip().upper()
            kind = "camera" if label.startswith("CAMERA") else "action" if label.startswith("ACTION") else \
                "sfx" if label.startswith("SOUND") else "dialogue"
            blocks.append((kind, _sec(b.group("s")), _sec(b.group("e"))))
        eps[int(m.group(1))] = blocks
    return eps


def check(data_dir: str, doc: dict):
    eps = _load(data_dir)
    errors, lines = [], []
    promos = doc.get("ranked_promos", [])
    prev = {}
    seen_borderline = False
    for i, p in enumerate(promos, 1):
        pid = p.get("promo_id", f"#{i}")
        if p.get("rank") != i:
            errors.append(f"{pid}: rank {p.get('rank')} but position {i}")
        group = bool(p.get("borderline"))
        if seen_borderline and not group:
            errors.append(f"{pid}: a clean promo is ranked below a borderline one")
        seen_borderline = seen_borderline or group
        if group in prev and p["total_score"] > prev[group]:
            errors.append(f"{pid}: score {p['total_score']} higher than the promo ranked above it")
        prev[group] = p["total_score"]
        total, last = 0.0, {}
        for j, seg in enumerate(p["segments"], 1):
            ep = seg["episode"]
            where = f"{pid} seg{j} E{ep}"
            if ep not in eps:
                errors.append(f"{where}: no such episode")
                continue
            s, e = _sec(seg["start_timestamp"]), _sec(seg["end_timestamp"])
            blocks = eps[ep]
            if s not in {b[1] for b in blocks}:
                errors.append(f"{where}: start {seg['start_timestamp']} is not a block start")
            if e not in {b[2] for b in blocks}:
                errors.append(f"{where}: end {seg['end_timestamp']} is not a block end")
            for kind, bs, be in blocks:
                if kind == "dialogue" and (bs < s < be or bs < e < be):
                    errors.append(f"{where}: cuts a dialogue line ({bs}-{be}) mid-way")
                    break
            if e <= s:
                errors.append(f"{where}: end not after start")
            dur = round(e - s, 2)
            if abs(seg.get("duration_seconds", dur) - dur) > 0.011:
                errors.append(f"{where}: duration_seconds {seg['duration_seconds']} != {dur}")
            if ep in last and s < last[ep]:
                errors.append(f"{where}: overlaps / precedes an earlier segment of the same episode")
            last[ep] = e
            total += dur
        total = round(total, 2)
        if abs(p["total_duration_seconds"] - total) > 0.011:
            errors.append(f"{pid}: total_duration_seconds {p['total_duration_seconds']} != {total}")
        if not 20 <= total <= 60:
            errors.append(f"{pid}: total {total}s outside 20-60 s")
        if sum(p["scoring"].values()) != p["total_score"]:
            errors.append(f"{pid}: scoring sums to {sum(p['scoring'].values())}, total_score {p['total_score']}")
        lines.append(f"{pid} rank {p['rank']:>2} score {p['total_score']:>3} {total:6.2f}s {len(p['segments'])} seg")
    return errors, lines


def verify_file(data_dir: str, promos_path: str) -> bool:
    with open(promos_path, encoding="utf-8") as f:
        doc = json.load(f)
    errors, lines = check(data_dir, doc)
    print(f"verify: {len(doc.get('ranked_promos', []))} promos checked against {data_dir}", file=sys.stderr)
    if errors:
        print("verify: FAILED", file=sys.stderr)
        for e in errors:
            print("  - " + e, file=sys.stderr)
        return False
    print("verify: all timestamps, durations and scores check out", file=sys.stderr)
    return True
