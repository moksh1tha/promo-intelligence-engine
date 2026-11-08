# Changelog

## v0.3.3: dedup prefers clean clips (2026-11-08)

- Dedup: of two overlapping clips, the one without borderline scores, then without review flags,
  is kept; only then the higher score wins. The local review checks now run before dedup
  (`pipeline.dedup`). The ranking order of the kept clips is unchanged. 2 tests added.
- Effect on v4, replayed from the cache with no API calls (`runs/gemini-batch-v4-dedup`): one swap.
  The flagged ep12→13 clip (82) gives way to the clean R03 clip (80, recommended). Recall 13/17 all,
  5/17 recommended (v4 as run: 12 and 4); 19 recommended, 35 need review.
- New data: reference picks or probes made for another show now stop with a clear message naming
  the flag to pass (`--reference ''`, `--probes ''`) instead of the error `3`; parse warnings are
  printed in the log. README: step-by-step guide for new scripts (section G) and a full description
  of outputs and raw output files (section H). 1 test added (60).
- The raw-output package now holds only the recommended run (gemini-batch-v3), as `raw_output/`
  and `raw_output.zip`: `final_output/`, `llm_responses/`, `intermediate/`, `logs/` and
  `replay_cache/` (the 17 cached replies a replay of the run reads, found by
  `tools/trace_cache.py`; 6 of them are earlier critic replies whose reviews v3 reused). The v3
  replay from the package no longer needs any file removed. `compare` reads the new layout.
- SOLUTION.md §7 rewritten: why recommended-only recall stays at 5/17 (flag and rejection counts
  from v4), what a stronger model can and cannot fix, and other levers. Still 2 pages.

## v0.3.2: v4 run, raw-log fix (2026-11-08)

- Ran `runs/gemini-batch-v4/` (arc planner v2 + `--gap-pass`, same models; 9 live requests, 5 cache
  hits). 54 promos (18 recommended / 36 need review), 40 episodes, recall 12/17 all and 4/17
  recommended (v3: 13 and 5). Of 7 new promos, 3 good, 3 borderline, 1 poor. v3 stays the headline
  result; see the run's `RESULTS.md`.
- Fixed (AI-written bug): the gap pass reused the stage names `repair_batch01` and
  `critic_batch01_01`, so its raw files overwrote the main pass's in v4's `raw/`. `Run.ask` now
  suffixes a repeated stage name (`_2`, `_3`); regression test added (57 tests). The two lost
  responses are intact in the response cache; v4's folder is kept as produced. A cache-only replay
  with the fix gives identical promos (`runs/gemini-batch-v4-replay`, 0 API calls).
- Known issue (fixed in v0.3.3): dedup kept the higher-scoring of two near-duplicates even when that
  replaced a clean clip with a flagged one (v4 lost reference R03 this way).
- SOLUTION.md: v3 made 4 live requests, not 3; 57 tests, not 55; probe scores are reused in v3/v4,
  not re-scored each run; `RECALL_STUDY.md` is in the raw-output package. Added the v4 results.
- `tools/build_submission.py`: includes v4 and `--final <run>`; copies each run's `RESULTS.md`.

## v0.3.1: submission preparation (2026-11-08)

- `tools/build_submission.py`: builds the raw-output package (`submission/`, `submission.zip`) by
  copying run files and every cached model exchange, SHA-256 verified, with a manifest and audit
  (`tools/submission_README.md`, `tools/submission_AUDIT.md`). Makes no model calls.
- Fixed: `live_model_calls` did not count requests answered by `--fallback-model`. The v3 manifest
  says 3; its raw folder shows 4. The stored manifest is unchanged; the code is fixed and tested.
- Replay: a cache miss on the main model now tries the fallback model's cached answer, the same way
  a quota error does. Replaying v3 reproduces its promos, rejections and plan exactly.
- `compare` accepts the package layout (`final_outputs/promos.json`, `run_manifest.json`); the review
  memo skips non-cache JSON files.
- Tests skip with a clear message when the confidential input pack is not in `data/`; a missing
  `data/` folder gives an error message instead of a traceback.
- Placeholders in `.env.example` are treated as "no key". Manifests record `code_version`.
- `.gitignore` excludes `data/`, `runs/`, `submission/` and the cache: they hold Kuku's confidential
  content and must not be published. README rewritten as a step-by-step guide.


## v0.3: recall study and better discovery (2026-11-08)

Study: `runs/gemini-batch-v3/RECALL_STUDY.md`. Improved run: `runs/gemini-batch-v3/` (original runs
unchanged). All 9 missed references had been proposed; they were lost to unreliable LLM repair,
single-score rejections and missing cross-episode setups.

### Added
- `repair.py`: deterministic edge repair. It moves only the outer edges of a clip, block by block, keeping
  the model's hook and cliffhanger, and re-points an off-screen hook or cliffhanger at the nearest beat on
  screen. Runs before the LLM repair; fixed 27 of 30 invalid candidates with no model call.
- Borderline critic policy (default; `--critic-policy strict` restores v0.2): one weak subjective
  dimension on a clip totalling 75 or more is kept with a `borderline_subjective` review flag instead of
  rejected. A misleading edit or low coherence goes to review. Borderline promos rank after clean ones.
- `memo.py`: reuses earlier critic reviews per clip when a clip shows exactly the same blocks in the
  same order (47 of 79 reused in v3). A clip whose blocks changed is reviewed again.
- `--gap-pass`: one broader scouting request per 10 episodes with no surviving clip, with a list of moment
  types (betrayals, recognition, relationship reveals, coerced confessions, interrupted reveals...).
- Arc planner v2 (default; `--arc-planner 1` for v0.2): sees each episode's first and last line and is
  asked for setup→payoff, recognition, continuation and betrayal-callback ideas. More ideas cost no extra
  requests in batch mode.
- `--fallback-model`: answers main-model requests after its daily quota runs out.
- `python3 -m promo_finder compare <runs...>`: one evaluation method for every run, with recall over all
  promos and over recommended-only, split coverage and review counts.
- 13 regression tests (55 in total; 56 after v0.3.1).

### Changed / fixed
- HTTP socket timeouts are retried (on Python 3.9 they are not `TimeoutError`); the default timeout is 300 s.
- The verifier checks score order within clean and borderline groups, and that clean promos come first.
- Probe verdicts are pass / review / reject; a known-bad clip counts as caught if it is not passed.

### Not yet run
- Arc planner v2 and the gap pass need about 10 new requests (counted by a cache-only dry run). They wait
  for the free-tier quota reset, so they are not in `gemini-batch-v3`.


## v0.2: audit and human-in-the-loop review (2026-11-08)

Triggered by an audit of the submitted run `runs/gemini-batch` (kept unchanged; audited copy in
`runs/gemini-batch-audited/`). No new model calls were made for the audit.

### Bugs found and fixed
- **Cliffhanger/hook blocks that are never on screen were accepted** (`candidates.py`). The validator
  measured the clip's end including CAMERA shots trimmed by the cut, so a shot starting exactly at
  the cut counted as "in the clip". Affected 5 of 31 promos (P007, P010, P011, P014, P030). The clip
  timeline now drops blocks visible for less than 0.3 s and clips displayed times to the cut. In new
  runs this is a validation error and triggers the existing repair step.
- **Displayed clip times could run past the clip's end** (same root cause). Reports and critic
  prompts now show only what is on screen.
- **A malformed critic score key became a silent 0** (`critic.py`). Gemini returned
  `"narrative_ coherence": 10`; P029 was shown with coherence 0/10 and total 74 instead of 84. Keys
  are now normalised; a score that is still missing is recorded and flagged for review and is never
  used to pass or reject. 1 of 114 critic reviews was affected; no rejected candidate depended on it.
- **Reports hid CAMERA lines**, making visual hooks look absent (P002, P011, P021 looked like they
  started at 3.5–6.5 s). Reports now show every on-screen block.

### Added
- `promo_finder/review.py`: local human-review flags, each labelled objective / approximate /
  subjective, with a suggested action (accept / inspect / repair / reject) and the relevant blocks:
  hook late or not on screen, hook is a camera shot only, hook/cliffhanger description not supported
  (approximate word overlap), cliffhanger not on screen (with a source-backed metadata fix), clip
  opens or ends on a split sentence, story order, cross-episode cut, timestamp mismatch, missing
  critic score, zero or low coherence, critic warnings, scout/critic disagreement.
- Every run now writes a review status per promo and a "Human review queue" section in `report.md`.
- `python3 -m promo_finder audit --run <run>`: re-checks a finished run from its saved files (no model
  calls), applies only source-supported metadata fixes, re-ranks by corrected scores (keeping the
  original rank and score), reconciles counts, and writes `<run>-audited/` with `audit_report.md`.
- Manifest: `model_calls` renamed `live_model_calls`; added `recorded_responses`, `counts_note` and
  `review_status_counts`, so cached re-runs are not mistaken for fresh API benchmarks.
- `runs/gemini-batch-audited/MANUAL_AUDIT.md`: AI-made manual audit of the top 10 and the
  cross-episode promos.
- 13 regression tests (42 in total).

### Not changed
- Original run folders, raw model outputs, source scripts, prompts used by the submitted run,
  provider abstraction, number of model calls.

## v0.1 (2026-11-08)
Initial pipeline, providers, batch mode, independent verifier, reference picks and critic probes.
