"""Compare finished runs with one methodology. No model calls.

  python3 -m promo_finder compare runs/gemini-batch-audited runs/gemini-batch-v3

For each run (its promos.json) it reports:
  * promos, split into recommended (no review flag) / needs review / not recommended
  * promos kept only because of the borderline policy
  * episodes represented
  * reference recall: a reference promo counts as found when one promo covers >= 50 % of its
    runtime (the same rule as every report). Reported for all promos and for recommended only,
    because a promo waiting for human review is not yet a confident recommendation.
  * "split coverage": references covered >= 50 % only by the union of several promos (shown
    separately, never added to recall)

The reference set is subjective (picked by an AI reading the scripts), so recall here is a
consistency check, not ground truth.
"""
import json
import os
from typing import Dict, List, Tuple

from . import timecode
from .fixtures import load_fixture_file
from .screenplay import load_show

Interval = Tuple[int, float, float]


def _intervals(p: dict) -> List[Interval]:
    return [(s["episode"], timecode.parse(s["start_timestamp"]), timecode.parse(s["end_timestamp"]))
            for s in p["segments"]]


def _overlap(a: List[Interval], b: List[Interval]) -> float:
    return sum(max(0.0, min(x1, y1) - max(x0, y0)) for e1, x0, x1 in a for e2, y0, y1 in b if e1 == e2)


def _find(run_dir: str, *names: str) -> str:
    """Accept a run folder (runs/<run>/promos.json) and the raw-output package layout
    (raw_output/final_output/promos.json, raw_output/logs/run_manifest.json)."""
    for n in names:
        for p in (os.path.join(run_dir, n), os.path.join(run_dir, "final_output", n),
                  os.path.join(run_dir, "final_outputs", n), os.path.join(run_dir, "logs", n)):
            if os.path.exists(p):
                return p
    return ""


def evaluate_run(run_dir: str, refs) -> Dict:
    with open(_find(run_dir, "promos.json"), encoding="utf-8") as f:
        doc = json.load(f)
    promos = doc["ranked_promos"]
    status = lambda p: (p.get("review") or {}).get("status", "recommended")  # noqa: E731
    rec = [p for p in promos if status(p) == "recommended"]
    rows, found_all, found_rec, split = [], 0, 0, 0
    for ref, item in refs:
        ri = ref.intervals()
        dur = ref.total_duration
        by = next((p["promo_id"] for p in promos if _overlap(ri, _intervals(p)) >= 0.5 * dur), None)
        by_rec = next((p["promo_id"] for p in rec if _overlap(ri, _intervals(p)) >= 0.5 * dur), None)
        union = sum(_overlap(ri, _intervals(p)) for p in promos)
        is_split = by is None and union >= 0.5 * dur
        found_all += by is not None
        found_rec += by_rec is not None
        split += is_split
        rows.append({"id": ref.id, "note": item.get("note", ""), "matched_by": by, "matched_by_recommended": by_rec,
                     "split_coverage": is_split})
    counts: Dict[str, int] = {}
    for p in promos:
        counts[status(p)] = counts.get(status(p), 0) + 1
    manifest = {}
    mpath = _find(run_dir, "manifest.json", "run_manifest.json")
    if mpath:
        with open(mpath, encoding="utf-8") as f:
            manifest = json.load(f)
    probes = manifest.get("probes") or []
    return {
        "run": run_dir, "promos": len(promos), "status_counts": counts,
        "borderline_kept": sum(1 for p in promos if p.get("borderline")),
        "episodes_represented": len({s["episode"] for p in promos for s in p["segments"]}),
        "recall_all": found_all, "recall_recommended": found_rec, "split_coverage": split, "references": len(refs),
        "probes_correct": sum(1 for p in probes if p.get("ok")) if probes else None, "probes": len(probes),
        "rows": rows,
    }


def compare(data_dir: str, run_dirs: List[str], reference: str) -> List[Dict]:
    show = load_show(data_dir)
    refs = load_fixture_file(show, reference, origin="reference")
    return [evaluate_run(r, refs) for r in run_dirs]


def to_markdown(results: List[Dict]) -> str:
    n = results[0]["references"] if results else 0
    L = ["| Metric | " + " | ".join(os.path.basename(r["run"].rstrip("/")) for r in results) + " |",
         "|---" * (len(results) + 1) + "|"]

    def row(label, fn):
        L.append(f"| {label} | " + " | ".join(fn(r) for r in results) + " |")
    row("Promos", lambda r: str(r["promos"]))
    row("… recommended (no review flag)", lambda r: str(r["status_counts"].get("recommended", 0)))
    row("… needs human review", lambda r: str(r["status_counts"].get("needs_review", 0)))
    row("… kept by the borderline policy", lambda r: str(r["borderline_kept"]))
    row("Episodes represented", lambda r: str(r["episodes_represented"]))
    row(f"Reference recall, all promos (of {n})", lambda r: f"{r['recall_all']} ({r['recall_all'] / n:.0%})")
    row(f"Reference recall, recommended only (of {n})",
        lambda r: f"{r['recall_recommended']} ({r['recall_recommended'] / n:.0%})")
    row("Covered only by several promos (not counted)", lambda r: str(r["split_coverage"]))
    row("Critic probes correct", lambda r: f"{r['probes_correct']}/{r['probes']}" if r["probes"] else "–")
    L.append("")
    L.append("| Reference | " + " | ".join(os.path.basename(r["run"].rstrip("/")) for r in results) + " |")
    L.append("|---" * (len(results) + 1) + "|")
    for i, ref in enumerate(results[0]["rows"] if results else []):
        cells = []
        for r in results:
            x = r["rows"][i]
            cells.append((x["matched_by"] or ("split" if x["split_coverage"] else "–")) +
                         (" (review)" if x["matched_by"] and not x["matched_by_recommended"] else ""))
        L.append(f"| {ref['id']} {ref['note'][:45]} | " + " | ".join(cells) + " |")
    return "\n".join(L)
