"""Replay a run from the cache and write the list of cache files it reads (direct hits + reused
clip reviews). Used by build_submission.py to package exactly what a replay needs. No API calls:

  python3 tools/trace_cache.py tools/replay_cache_<run>.json run --provider replay:gemini <same options as the run> --out <tmp folder>
"""
import json, os, sys
sys.path.insert(0, os.getcwd())
from promo_finder import llm, memo, cli
used = set()
orig_get = llm.LLM._cache_get
def get(self, key):
    r = orig_get(self, key)
    if r is not None:
        used.add(key + ".json")
    return r
llm.LLM._cache_get = get
orig_init = memo.CritiqueMemo.__init__
def init(self, cache_dir, provider, model, effort, show=None):
    orig_init(self, cache_dir, provider, model, effort, show)
    # rebuild key -> file with the same last-wins order
    self._src = {}
    import glob
    for f in glob.glob(os.path.join(cache_dir or "", "*.json")):
        try: d = json.load(open(f, encoding="utf-8"))
        except Exception: continue
        if not isinstance(d, dict) or (d.get("provider"), d.get("model"), d.get("effort")) != (provider, model, effort): continue
        p = d.get("prompt", "")
        if "=== CLIP " not in p: continue
        try: payload = llm.extract_json(d.get("response", ""))
        except Exception: continue
        ids = {str(r.get("id")) for r in (payload.get("reviews", []) if isinstance(payload, dict) else [])}
        for cid, text in memo.CLIP_RE.findall(p):
            k = memo._parse_clip_text(show, text)
            if k and cid in ids: self._src[k] = os.path.basename(f)
memo.CritiqueMemo.__init__ = init
orig_mget = memo.CritiqueMemo.get
def mget(self, cand):
    r = orig_mget(self, cand)
    if r is not None:
        used.add(self._src[memo.clip_block_key(self.show, cand)])
    return r
memo.CritiqueMemo.get = mget
out = sys.argv[1]; sys.argv = ["promo_finder"] + sys.argv[2:]
try:
    cli.main()
except SystemExit:
    pass
json.dump(sorted(used), open(out, "w"), indent=1)
print("cache files used:", len(used))
