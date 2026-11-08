# Promo Finder

Finds promotional clips (20–60 s "promos") in timestamped microdrama screenplays. For every promo it
gives the episode and the start/end timestamps of every segment, the hook, the cliffhanger, a
one-line reason, a score and a rank. It also lists every rejected candidate with the reason, and
flags uncertain picks for a human editor. Built for the Kuku AI Generalist assignment (Problem 1).

- Design, results and limitations: [SOLUTION.md](SOLUTION.md) (2 pages; also as `SOLUTION.pdf`)
- Change history: [CHANGELOG.md](CHANGELOG.md)
- Running it on your own scripts: section **G**; what every output and raw output file holds: section **H**
- Raw output of the recommended 50-episode run: shared separately as `raw_output.zip` (link in
  the submission form), because it contains Kuku's confidential scripts

## Quick start

```bash
git clone https://github.com/<your-username>/promo-finder.git
cd promo-finder
# copy the 50 episode .txt files of the input pack into data/ (not in this repo: confidential)
python3 -m unittest discover -s tests            # 60 tests, no API key needed
python3 -m promo_finder run --provider heuristic --out runs/heuristic-baseline   # offline, no key
```

With a free Gemini key in `.env`, the submitted configuration is:

```bash
python3 -m promo_finder run --provider gemini --model gemini-3.5-flash --critic-model gemini-3-flash-preview --fallback-model gemini-3-flash-preview --batch --arc-planner 1 --arc-ideas 5 --out runs/my-run
```

## How it works (one paragraph)

`promo_finder` parses each script into numbered blocks (dialogue, action, sound, camera). An LLM
proposes clips **by block number**, so timestamps are always copied from the script, never invented.
Code then checks the hard rules (20–60 s, hook in the first 3 s, ends on the cliffhanger, no
dialogue cut mid-line, story order) and repairs edge problems itself before asking the LLM. A second,
blind LLM call scores each clip as a first-time viewer would. The tool ranks the passing clips,
attaches human-review flags and writes the outputs. An independent verifier then re-checks every
timestamp against the scripts. Details are in [SOLUTION.md](SOLUTION.md).

```
promo_finder/   screenplay.py (parse) · scout.py (LLM clip finding) · candidates.py (rules) · repair.py
                (code repair) · critic.py (LLM review) · review.py (human-review flags) · pipeline.py
                (orchestration) · llm.py (model providers, cache) · prompts.py · report.py · verify.py
                · audit.py · compare.py · memo.py · cli.py (entry point: python3 -m promo_finder)
tests/          60 unit and end-to-end tests
eval/           reference_picks.json (17 hand-picked promos for recall) · critic_probes.json (6 test clips)
tools/          build_submission.py (raw-output package) · trace_cache.py (cache files a replay needs)
                · check_pages.py (SOLUTION.md page count)
```

## A. Prerequisites

- **Python 3.9 or newer** (tested on 3.9.6). Nothing else is needed for the offline baseline,
  the free providers, the tests, or inspecting results.
- **The input pack:** the 50 episode `.txt` files go in `data/`. They are not in this repository.
- **For a model run, one of:**
  - a **free Gemini API key** (Google AI Studio). Recommended; the submitted runs used it.
  - a Groq or OpenRouter key, or a local Ollama server (any OpenAI-compatible API works);
  - an Anthropic API key, plus the `anthropic` package (`requirements.txt`).
- Optional: Google Chrome, only for `tools/check_pages.py`.

## B. Clone the repository

```bash
git clone https://github.com/<your-username>/promo-finder.git
```

```bash
cd promo-finder
```

Then copy the input pack's 50 `.txt` files into `data/` (e.g. `data/episode-001-Episode 1671703.txt`).

## C. Create and activate a virtual environment (optional but recommended)

macOS / Linux:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

Windows (PowerShell or cmd; not tested by the author):

```
py -m venv .venv
.venv\Scripts\activate
```

On Windows, type `python` wherever this README says `python3`.

## D. Install dependencies

The core uses only the Python standard library. Only `--provider anthropic` needs a package:

```bash
pip install -r requirements.txt
```

## E. Configure environment variables

```bash
cp .env.example .env
```

(Windows: `copy .env.example .env`). Open `.env` and replace the placeholder for the provider you
will use, e.g. `GEMINI_API_KEY=AI...`. The tool reads `.env` from the folder you run it in, so run
commands from the project root. Real environment variables take precedence over `.env`.

| Provider (`--provider`) | Variable in `.env` | Default model |
|---|---|---|
| `gemini` (free tier) | `GEMINI_API_KEY` | `gemini-flash-latest` |
| `groq` | `GROQ_API_KEY` | none: pass `--model` |
| `openrouter` | `OPENROUTER_API_KEY` | none: pass `--model` |
| `ollama` (local) | none | none: pass `--model` |
| `openai` (any compatible API) | `OPENAI_API_KEY` (+ `--base-url` or `OPENAI_BASE_URL`) | none: pass `--model` |
| `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |

**Never commit `.env` or a real key.** `.env` is in `.gitignore`. To list the models your key can use:

```bash
python3 -m promo_finder models --provider gemini
```

## F. Run the prototype

All runs read `data/` (change with `--data <folder>`) and write to the folder given by `--out`.

**Submitted configuration** (free Gemini, batch mode: about 16 requests for 50 episodes, within the
free 20 requests per model per day):

```bash
python3 -m promo_finder run --provider gemini --model gemini-3.5-flash --critic-model gemini-3-flash-preview --fallback-model gemini-3-flash-preview --batch --arc-planner 1 --arc-ideas 5 --out runs/my-run
```

**Offline baseline** (no model, no key, about 1 second):

```bash
python3 -m promo_finder run --provider heuristic --out runs/heuristic-baseline
```

**Claude** (paid; one request per episode and per clip, no batching needed):

```bash
python3 -m promo_finder run --provider anthropic --out runs/claude
```

**Other options** (all in `python3 -m promo_finder run --help`):

| Option | Effect |
|---|---|
| `--episodes 1-5` | process a subset (try this first) |
| `--model`, `--critic-model` | model that finds clips / model that reviews them |
| `--batch` | 10 episodes / 15 clips per request, for low daily quotas |
| `--fallback-model` | answers main-model requests once its daily quota is used up |
| `--critic-policy borderline\|strict` | borderline (default): one weak subjective score on a clip totalling 75 or more goes to human review instead of rejection |
| `--arc-planner 1\|2`, `--arc-ideas N`, `--gap-pass` | cross-episode planning version, number of ideas, extra look at episodes with no clip |
| `--rpm N`, `--workers N` | requests per minute, parallel requests |
| `--cache <dir>` | response cache folder (default `.cache/llm`) |

**Resume an interrupted run:** run the same command again. Every model response is cached in
`.cache/llm/`, so finished requests are not repeated. This is also how to continue after a free-tier
daily quota runs out.

**Use cached results only (no API calls):** use `--provider replay:<provider>` with the same models
and options. Any request that is not in the cache fails, so this also tells you how many new
requests a configuration would need before you spend any:

```bash
python3 -m promo_finder run --provider replay:gemini --model gemini-3.5-flash --critic-model gemini-3-flash-preview --fallback-model gemini-3-flash-preview --batch --arc-planner 1 --arc-ideas 5 --out runs/replay
```

**Check, audit and compare runs** (no API calls):

```bash
python3 -m promo_finder verify --promos runs/my-run/promos.json
```

```bash
python3 -m promo_finder audit --run runs/my-run
```

```bash
python3 -m promo_finder compare runs/heuristic-baseline runs/my-run
```

`verify` re-checks every timestamp with an independent parser. `audit` writes `runs/my-run-audited/`
with human-review flags and source-backed fixes. `compare` scores runs against the 17 reference
picks with one method.

## G. Run it on new scripts (your own show)

The tool works on any show whose scripts follow the input pack's format. Step by step:

**1. Check the file format.** One UTF-8 `.txt` file per episode. The file name must contain
`episode-<number>` (e.g. `episode-001-Pilot.txt`), or the file must start with
`EPISODE NUMBER: Episode 1`. Inside, every line of the script is a block like this:

```
EPISODE NUMBER: Episode 1
EPISODE LOGLINE: One sentence about the episode (used for cross-episode planning).
CONTINUITY NOTES: Optional.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. BALLROOM – NIGHT
Scene start: 00:00.00
Scene end: 01:30.00

Actions and dialogue:

**ACTION:**
Start: (00:00.00) | End: (00:04.00)
A woman bursts into the ballroom, soaked from the storm.

**MARA:**
Start: (00:04.00) | End: (00:08.00)
Stop the wedding!
```

Labels `CAMERA`, `ACTION` and `SOUND FX`/`SFX` are treated as such; any other label is a speaker,
e.g. `**LADY JULIANA (Breathless):**`. Times are `mm:ss.cc` (or `hh:mm:ss.cc`) from the start of
the episode video. Every promo timestamp is copied from these lines, so they must match the video.

**2. Put the files in a folder,** e.g. `my_show/`. Keep it out of git if the scripts are private
(`data/` is already ignored).

**3. Check that every file parses** (offline, no key, a few seconds):

```bash
python3 -m promo_finder run --provider heuristic --data my_show --reference '' --probes '' --out runs/my-show-check
```

The log starts with `Parsed N episodes, M blocks`; check that N is your episode count. Problems in
a file (no blocks found, a block that ends before it starts, a file name and header that disagree)
are printed as `warning: episode N (file): ...` and saved under `parse_warnings` in
`runs/my-show-check/manifest.json`.

**4. Why `--reference '' --probes ''`:** `eval/reference_picks.json` and `eval/critic_probes.json`
are hand-made for the assignment's show. On other scripts the tool skips them with a message saying
so; passing `''` skips them silently. To measure recall on your show, write your own file in the
same format as `eval/reference_picks.json` and pass it with `--reference`.

**5. Try 2 episodes with a model first** (about 4 requests):

```bash
python3 -m promo_finder run --provider gemini --model gemini-3.5-flash --critic-model gemini-3-flash-preview --batch --data my_show --episodes 1-2 --reference '' --probes '' --out runs/my-show-test
```

**6. Run the whole show** (about 16 requests per 50 episodes in batch mode):

```bash
python3 -m promo_finder run --provider gemini --model gemini-3.5-flash --critic-model gemini-3-flash-preview --fallback-model gemini-3-flash-preview --batch --gap-pass --data my_show --reference '' --probes '' --out runs/my-show > runs/my-show.log 2>&1
```

`--gap-pass` gives episodes without a clip a second look (2–3 extra requests). Any provider from
section E works; with Claude, drop `--batch` and the Gemini model names.

**7. Check and read the result** (no API calls):

```bash
python3 -m promo_finder verify --data my_show --promos runs/my-show/promos.json
```

Then open `runs/my-show/report.md`: the ranked promos, the publishing plan and the human-review
queue (section H explains every file). If the quota runs out, run the step 6 command again: it
resumes from the cache.

## H. Outputs and raw outputs

Each run folder (`--out`) contains:

| File | Contents |
|---|---|
| `promos.json` | **Final output.** `ranked_promos`: for each promo the segments (episode, `HH:MM:SS.cc` start/end, block range), hook and cliffhanger (timestamp, source line, description), standalone context, why it works, scores, publishing priority, and `review` (status `recommended` / `needs_review`, suggested action, flags with evidence). `rejected_candidates`: every clip that did not make it, with `stage_rejected` and `reason_for_rejection`. Also `publishing_plan` and `analysis_summary` (episodes analysed, counts). |
| `report.md` | The same in readable form: ranking, publishing plan, **human review queue**, each clip as the viewer sees it, QC failure table, critic probes, reference recall |
| `candidates_all.jsonl` | **Intermediate output**, one JSON line per candidate (accepted, rejected, repaired, duplicate): `origin`, `segments`, `hook`, `cliffhanger`, validation `errors`/`warnings`, `rejection_reasons`, `duplicate_of`, `repaired_from`, the scout's `self_scores` and the critic's `scores` and `critique` |
| `raw/*.json` | **Raw outputs: every model request of the run, unedited.** One file per request, see below |
| `raw/_system_prompts.json` | The full system prompts (`STRATEGIST_SYSTEM`, `CRITIC_SYSTEM`, `RANKING_SYSTEM`) that raw files refer to by name |
| `manifest.json` | Models, prompt and code version, counts at each stage, live calls vs cache hits, `task_failures`, `parse_warnings`, probe results, reference recall |
| console output | Progress log. Keep it with `> runs/<run>.log 2>&1` |

**Raw output files.** Each `raw/<stage>.json` holds exactly what was sent and received:

```
{"stage": "critic_batch01_01", "model": "gemini:gemini-3-flash-preview (effort=low)", "seconds": 15.18,
 "system": "CRITIC_SYSTEM", "prompt": "<full prompt text>", "response": "<the model's reply, unedited>"}
```

| File name | Pipeline stage |
|---|---|
| `scout_ep001-010.json` (batch) / `scout_ep001.json` | clip finding for those episodes |
| `arc_planner.json`, `arc_build_batch.json` / `arc_buildNN.json` | cross-episode ideas, then their clips |
| `gap_scout_epNNN-MMM.json` | `--gap-pass`: second look at episodes without a clip |
| `repair_batchNN.json` / `repair_<id>.json` | LLM repair of clips that failed the rules |
| `critic_batchNN_MM.json` / `critic_<id>.json` | blind critic reviews |
| `probe_batchNN_MM.json` | critic calibration probes |
| `ranking.json` | ranking rationale and publishing plan |

A name used twice in one run gets a suffix (`repair_batch01_2.json`). `seconds` near 0 means the
reply came from the cache (`.cache/llm/`), answered live by an earlier run with the same prompt.
The cache holds every exchange ever made (`provider`, `model`, `prompt`, `response`); deleting it
only means the next run asks again. To read raw outputs quickly:

```bash
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['model'], d['seconds'], 's'); print(d['response'])" runs/my-show/raw/ranking.json
```

## I. Run tests

```bash
python3 -m unittest discover -s tests -v
```

The 60 tests need the input pack in `data/`; without it they are skipped with a message. They cover
script parsing, every validation rule (with deliberately broken clips: late or weak hooks, misleading
or off-screen cliffhangers, missing context, mid-dialogue cuts, wrong durations and timestamps, story
order), code repair, the critic policies, human-review flags, the independent verifier (including a
tampered output), provider retries and quota handling against a local fake server, cache accounting,
batch mode within free quotas, and end-to-end runs with a fake model.

## J. Inspect or reproduce the submitted run

**Without an API key:** open the raw-output package (`raw_output.zip`, shared by link). It holds
the recommended run, gemini-batch-v3. Start with `raw_output/final_output/report.md` and
`raw_output/AUDIT.md`; every model prompt and response is under `raw_output/llm_responses/`. With
this repo and the input pack in `data/` you can also:

```bash
python3 -m promo_finder verify --promos raw_output/final_output/promos.json
```

```bash
python3 -m promo_finder compare raw_output
```

```bash
python3 -m promo_finder run --provider replay:gemini --model gemini-3.5-flash --critic-model gemini-3-flash-preview --fallback-model gemini-3-flash-preview --batch --arc-planner 1 --arc-ideas 5 --cache raw_output/replay_cache --out runs/replay-v3
```

`compare` scores the run against the 17 reference picks. The last command rebuilds the run from
the package's `replay_cache/` with **zero API calls**; its ranked promos, rejected candidates and
publishing plan are identical to the package's. To rebuild the package from your own runs:
`python3 tools/build_submission.py` (`--run <run>` for another run).

**Fresh run (optional, uses your quota):** the "Submitted configuration" command in section F. Free
models do not answer identically twice, so expect similar but not identical promos.

## K. Troubleshooting

| Problem | Fix |
|---|---|
| `error: --provider gemini needs an API key: set GEMINI_API_KEY in your shell or in .env` | Create `.env` (section E), replace the placeholder, and run from the project root. |
| `no .txt screenplays found in data` / tests "skipped" | Copy the input pack's `.txt` files into `data/`, or pass `--data <folder>`. |
| `pip install anthropic  (needed for --provider anthropic)` | `pip install -r requirements.txt`, or use a free provider. |
| `--model is required for --provider groq` | List models with `python3 -m promo_finder models --provider groq` and pass one with `--model`. |
| "quota ran out during this run" / HTTP 429 `RESOURCE_EXHAUSTED` | Free Gemini allows about 20 requests per model per day. Use `--batch`, add `--fallback-model <another model>`, or wait for the reset and run the **same command** again (it resumes from the cache). |
| HTTP 503 "high demand", or slow responses | Retried automatically. Try another model (`models` command), lower `--workers`, or raise the timeout: `PROMO_HTTP_TIMEOUT=600`. |
| A replay run reports `replay: no cached response` | The cache has no answer for that prompt: the models, options, code or cache folder differ from the original run. |
| `python3: command not found` (Windows) | Use `python` or `py`. |
