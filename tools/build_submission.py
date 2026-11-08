"""Build the raw-output package for ONE run (default: the headline run). Makes no model calls.

  python3 tools/build_submission.py                          # packages runs/gemini-batch-v3
  python3 tools/build_submission.py --run gemini-batch-v4    # packages another run

Writes raw_output/ and raw_output.zip:

  raw_output/
    final_output/       promos.json, report.md (+ the run's study notes, if any)
    llm_responses/      every model request of the run, unedited, sorted by pipeline stage
    intermediate/       candidates_all.jsonl: every candidate the run produced
    logs/               console log and the run's manifest
    replay_cache/       exactly the cached model replies a cache-only replay of this run reads
                        (the run's own replies + earlier replies whose reviews it reused)
    README.md, AUDIT.md, SOLUTION.pdf, manifest.json (SHA-256 of every file)

Files are copied, never edited; every copy is checked against its source with SHA-256.
replay_cache/ comes from tracing a cache-only replay (tools/trace_cache.py); its file list is
read from tools/replay_cache_<run>.json.

The package contains Kuku's confidential input content (prompts embed the screenplays, outputs
quote dialogue): share it with the evaluators by link, do not publish it.
"""
import glob
import hashlib
import json
import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "raw_output"
OUT = os.path.join(ROOT, NAME)

STAGES = [  # (folder, filename prefixes)
    ("candidate_generation", ("scout_ep",)),
    ("candidate_generation_gap_pass", ("gap_scout_",)),
    ("cross_episode", ("arc_planner", "arc_build")),
    ("repair", ("repair",)),
    ("critic", ("critic_",)),
    ("critic_probes", ("probe_",)),
    ("ranking", ("ranking",)),
    ("system_prompts", ("_system_prompts",)),
]


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src: str, dst: str, files: list):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    if sha(src) != sha(dst):
        sys.exit(f"copy mismatch: {src}")
    files.append({"path": os.path.relpath(dst, OUT), "source": os.path.relpath(src, ROOT), "sha256": sha(dst),
                  "bytes": os.path.getsize(dst)})


def stage_of(name: str) -> str:
    for folder, prefixes in STAGES:
        if name.startswith(prefixes):
            return folder
    return "other"


def main():
    run = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else "gemini-batch-v3"
    src = os.path.join(ROOT, "runs", run)
    if not os.path.isdir(src):
        sys.exit(f"no such run: runs/{run}")
    cache_list = os.path.join(ROOT, "tools", f"replay_cache_{run}.json")
    if not os.path.exists(cache_list):
        sys.exit(f"missing {cache_list}: trace a replay of the run first (tools/trace_cache.py)")
    for old in (OUT, os.path.join(ROOT, "submission")):   # "submission" = the old multi-run layout
        if os.path.exists(old):
            shutil.rmtree(old)
    for old_zip in ("submission.zip", NAME + ".zip"):
        if os.path.exists(os.path.join(ROOT, old_zip)):
            os.remove(os.path.join(ROOT, old_zip))
    os.makedirs(OUT)
    files, stage_counts = [], {}

    for f in sorted(glob.glob(os.path.join(src, "raw", "*.json"))):
        st = stage_of(os.path.basename(f))
        stage_counts[st] = stage_counts.get(st, 0) + 1
        copy(f, os.path.join(OUT, "llm_responses", st, os.path.basename(f)), files)
    for name, sub in [("promos.json", "final_output"), ("report.md", "final_output"),
                      ("RECALL_STUDY.md", "final_output"), ("comparison.md", "final_output"),
                      ("RESULTS.md", "final_output"), ("candidates_all.jsonl", "intermediate"),
                      ("manifest.json", "logs")]:
        p = os.path.join(src, name)
        if os.path.exists(p):
            copy(p, os.path.join(OUT, sub, "run_manifest.json" if name == "manifest.json" else name), files)
    log = os.path.join(ROOT, "runs", run + ".log")
    if os.path.exists(log):
        copy(log, os.path.join(OUT, "logs", run + ".log"), files)

    # Cached replies a replay of this run reads; mark which are the run's own requests.
    own = set()
    for f in glob.glob(os.path.join(src, "raw", "*.json")):
        if not f.endswith("_system_prompts.json"):
            with open(f, encoding="utf-8") as fh:
                own.add(json.load(fh)["response"])
    with open(cache_list, encoding="utf-8") as fh:
        needed = json.load(fh)
    index = []
    for name in needed:
        f = os.path.join(ROOT, ".cache", "llm", name)
        copy(f, os.path.join(OUT, "replay_cache", name), files)
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        index.append({"file": name, "model": f"{d['provider']}:{d['model']}",
                      "source": "this run's request (also in llm_responses/)" if d["response"] in own
                      else "earlier run's critic reply whose clip reviews this run reused"})
    with open(os.path.join(OUT, "replay_cache", "_index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)

    with open(os.path.join(src, "manifest.json"), encoding="utf-8") as fh:
        rm = json.load(fh)
    with open(os.path.join(src, "promos.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    manifest = {
        "package": "Promo Finder raw output (Kuku AI Generalist assignment, Problem 1)",
        "confidentiality": "Contains Kuku's input-pack content (prompts embed the screenplays). Share by link only.",
        "how_built": "python3 tools/build_submission.py (copies only; every copy SHA-256 verified against its source)",
        "run": run,
        "models": {k: rm.get(k) for k in ("model", "critic_model", "fallback_model") if k in rm},
        "episodes_analyzed": len(doc["analysis_summary"]["episodes_analyzed"]),
        "promos": len(doc["ranked_promos"]),
        "rejected_candidates": len(doc["rejected_candidates"]),
        "llm_responses_by_stage": stage_counts,
        "replay_cache": {"entries": len(index),
                         "own_requests": sum(1 for i in index if i["source"].startswith("this run")),
                         "reused_from_earlier_runs": sum(1 for i in index if i["source"].startswith("earlier"))},
        "files": files,
    }
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    for doc_name in ("README.md", "AUDIT.md"):
        shutil.copy2(os.path.join(ROOT, "tools", "submission_" + doc_name), os.path.join(OUT, doc_name))
    if os.path.exists(os.path.join(ROOT, "SOLUTION.pdf")):
        shutil.copy2(os.path.join(ROOT, "SOLUTION.pdf"), os.path.join(OUT, "SOLUTION.pdf"))

    zpath = os.path.join(ROOT, NAME + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, names in os.walk(OUT):
            for n in sorted(names):
                full = os.path.join(dirpath, n)
                z.write(full, os.path.join(NAME, os.path.relpath(full, OUT)))
    print(f"{NAME}/: run {run}; {len(files)} files copied and verified; {len(index)} replay-cache entries; "
          f"zip: {zpath} ({os.path.getsize(zpath) // 1024} KB)")


if __name__ == "__main__":
    main()
