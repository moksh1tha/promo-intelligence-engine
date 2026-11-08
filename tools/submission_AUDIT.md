# Audit: raw output package (gemini-batch-v3)

Built by `python3 tools/build_submission.py` on 2026-11-08 from `runs/gemini-batch-v3/` and the
response cache. Files were **copied, never edited**: every file's SHA-256 matches its source
(listed in `manifest.json`). No model was called to build it.

This package holds only the recommended run. Earlier and later runs (the first submitted run v2,
the v4 experiment and others) are kept by the author; their results are summarised in
`SOLUTION.pdf` §4.

## 1. Checks performed

| Check | Result |
|---|---|
| All 50 episodes processed | ✅ `final_output/promos.json` lists episodes 1–50 in `analysis_summary.episodes_analyzed`; the 5 prompts in `llm_responses/candidate_generation/` contain all 50 episode scripts (3,727 blocks). |
| Every model request of the run included | ✅ All 11 requests the run made or served from the cache are in `llm_responses/`. |
| Responses unedited | ✅ Each file in `llm_responses/` was compared with the cache entry for the same model and prompt: all 11 responses are identical. Copies are SHA-256 verified. |
| Nothing fabricated | ✅ Every response came from a model call made by the tool (`promo_finder/pipeline.py` writes each file immediately after the call). None was written by hand or reconstructed. |
| Accepted and rejected candidates included | ✅ `final_output/promos.json` has 48 ranked promos **and** 61 `rejected_candidates` with reasons. `intermediate/candidates_all.jsonl` holds all 109 candidate records. |
| Final output from the same run as the responses | ✅ `promos.json`, `report.md`, `candidates_all.jsonl`, the manifest and the response files were written by the same invocation. Every promo and rejected ID matches a record in `candidates_all.jsonl` with the same block ranges. |
| Timestamps correct | ✅ `promo_finder verify` (independent parser) passes: 48 promos. |
| Reproducible without an API key | ✅ Replaying from `replay_cache/` rebuilds the run with 0 API calls; ranked promos, rejected candidates and publishing plan are identical. Command in `README.md`. |
| No secrets | ✅ No API key, `.env` file or credential in the package. Scanned for the key's value and for the `AIza` key prefix. |

## 2. Run manifest accuracy

| Manifest says | Response files show | Verdict |
|---|---|---|
| **3** live + 7 cache hits | 11 responses, 7 with ~0 s, so **4 live** | ❌ **Under-counted by 1.** The ranking request was answered by the fallback model, which the counter did not include. The code is fixed (`_usage`, regression test); the run's own manifest is kept unchanged. |

The run was made by code v0.3; it reproduces exactly with today's code (§1).

## 3. Missing or partial (known, not reconstructed)

1. **Repair request:** the run's single LLM repair request timed out, so there is no response; 3 candidates were lost.
2. **Reused reviews:** 47 of 79 clip reviews and the 6 probe reviews were reused from critic replies of earlier runs (same model, identical clips), so the run made no request for them. Those 6 replies are in `replay_cache/` (`_index.json`: "earlier run's critic reply"), not in `llm_responses/`, because this run did not send them.
3. **Ranking:** the main model's ranking request hit the free daily quota (HTTP 429); the fallback model's answer is the one included.
4. **Log:** `logs/gemini-batch-v3.log` covers this invocation only. The 7 cached replies were first produced live by an earlier invocation of the same configuration; their prompts and responses are in `llm_responses/` with `seconds` near 0.
