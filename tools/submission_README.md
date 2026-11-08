# Promo Finder: raw output (Kuku AI Generalist assignment, Problem 1)

Unedited output of the recommended Promo Finder run (**gemini-batch-v3**) on the **full input pack:
all 50 episodes**. Code and run instructions are in the GitHub repository; the design, results and
limitations are in `SOLUTION.pdf`.

> **Confidential.** Prompts embed Kuku's screenplays and outputs quote dialogue. This package is
> shared only with the evaluators; it is not published anywhere.

## Assignment checklist
| Required | Where |
|---|---|
| Original LLM responses | `llm_responses/<stage>/*.json`: full prompt and unedited reply of every model request of the run |
| Intermediate candidates | `intermediate/candidates_all.jsonl`: every candidate (accepted, rejected, repaired, duplicate) with validation errors and critic output |
| Logs | `logs/gemini-batch-v3.log` (console log) and `logs/run_manifest.json` (models, counts, live calls vs cache hits, failures) |
| Final JSON | `final_output/promos.json`: ranked promos and every rejected candidate with its reason |

## Contents
| Folder / file | Contents |
|---|---|
| `final_output/promos.json` | 48 ranked promos: segments (episode, start/end timestamps), hook, cliffhanger, standalone context, why it works, scores, review status and flags; plus 61 rejected candidates with the stage and reason |
| `final_output/report.md` | Readable version: ranking, publishing plan, human-review queue, each clip as the viewer sees it, QC failure table, critic probes, reference recall |
| `final_output/RECALL_STUDY.md`, `comparison.md` | Why the first submitted run missed references, and how this run compares with it |
| `llm_responses/` | 11 model requests: `candidate_generation` (5, 10 episodes each), `cross_episode` (2), `critic` (3), `ranking` (1); `system_prompts/` holds the system prompts the files name |
| `intermediate/candidates_all.jsonl` | 109 candidate records |
| `logs/` | Console log and the run's manifest |
| `replay_cache/` | The 17 cached replies a cache-only replay of this run reads (`_index.json` says which is which): the run's own 11 requests, plus 6 critic replies from earlier runs whose clip reviews this run reused |
| `AUDIT.md` | What was checked, what is missing and why |
| `manifest.json` | SHA-256 of every file, per-stage counts |

### Reading a raw LLM response
Each `llm_responses/<stage>/<name>.json` is one model request, exactly as sent and received:
```
{"stage": "critic_batch01_01", "model": "gemini:gemini-3-flash-preview (effort=low)", "seconds": 27.55,
 "system": "CRITIC_SYSTEM", "prompt": "<full prompt>", "response": "<the model's reply, unedited>"}
```
`seconds` near 0 means the reply came from the response cache: the same request had been answered
live by an earlier run of the same configuration (here: the 5 clip-finding and 2 cross-episode
requests). The 3 critic requests and the ranking were live. To print one:
```bash
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['model'], d['seconds'], 's'); print(d['response'])" raw_output/llm_responses/ranking/ranking.json
```

## Check or reproduce without an API key (needs the repo and the input pack in `data/`)
```bash
python3 -m promo_finder verify --promos raw_output/final_output/promos.json
python3 -m promo_finder run --provider replay:gemini --model gemini-3.5-flash --critic-model gemini-3-flash-preview --fallback-model gemini-3-flash-preview --batch --arc-planner 1 --arc-ideas 5 --cache raw_output/replay_cache --out runs/replay-v3
```
The second command rebuilds the run from `replay_cache/` with **zero API calls**; its ranked promos,
rejected candidates and publishing plan are identical to `final_output/promos.json`.
