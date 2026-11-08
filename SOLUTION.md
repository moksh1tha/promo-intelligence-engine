# Promo Finder: solution

## 1. What it does
Reads timestamped screenplays and returns ranked 20–60 s promos (segments, hook, cliffhanger, rationale, score, publishing plan) plus every rejected candidate with the reason. It only selects moments. See `README.md` and `CHANGELOG.md`.

## 2. How it works (five stages)
1. **Parse**: scripts become numbered blocks (dialogue, ACTION, SFX, CAMERA) with source times (3,727 blocks).
2. **Scout (LLM)**: proposes clips per episode, plus cross-episode ideas planned from the loglines. **It picks block numbers, never timestamps**; the tool copies times from the script, so timestamp hallucination is impossible by construction. It can still misdescribe what it picked, hence stages 3–5.
3. **Validate (code)**: 20–60 s; hook starts in the first 3 s; clip ends ≤ 4 s after the cliffhanger; hook and cliffhanger actually on screen; no dialogue cut mid-line; segments in story order. A failing clip is first **repaired by trimming its edges in code** (it keeps the model's hook and cliffhanger), and only then gets one LLM repair attempt. Every change is logged.
4. **Blind critic (LLM)**: a second prompt sees only the clip, as a first-time viewer would, and scores the 100-point rubric. It flags weak hook, missing context, predictable ending, low curiosity and misleading edit. Its scores replace the scout's own. Clear failures are rejected. **One weak subjective score on a clip totalling 75 or more goes to human review instead of rejection.** Reviews of identical clips are reused from the cache.
5. **Rank and review**: dedups, ranks, writes a publishing plan and runs local **human-review checks**. An **independent verifier** (own parser) then re-checks every timestamp, duration and score against the raw scripts.

## 3. Validation and the human in the loop
Free LLMs are imperfect, so the tool states its uncertainty. Each review flag names its evidence and suggests **accept / inspect / repair / reject**; subjective concerns never auto-reject, and an editor decides.

| Kind | Checks | Basis |
|---|---|---|
| Objective | hook late/off screen; cliffhanger off screen; camera-only hook; split sentence; story order; displayed vs source times; missing or zero critic score | exact, from blocks |
| Approximate | hook/cliffhanger description not supported by the clip | word overlap; can only ask for a look |
| Subjective | critic flags; low coherence; scout vs critic disagree by ≥ 15 points | LLM judgement |

`audit` re-checks a finished run and `compare` scores any runs with one method; both make **no model calls**. Six known good/bad clips check the critic (v3 reuses earlier scores for them). 60 tests cover broken clips (weak hooks, misleading endings, missing context, mid-line cuts), tampered outputs, quota errors and cache accounting.

## 4. Results and the recall study
**The raw output holds v3, the final run.** v2 (first version) and v4 (an experiment) were development runs, shown only for comparison: same models (free Gemini, batch mode) and match rule (a promo covers ≥ 50 % of a reference).

| | v2 (first version) | **v3 (final, in raw output)** | v4 (experiment) |
|---|---|---|---|
| Promos: recommended / needs review | 31: 14 / 17 | 48: **17 / 31** | 54: 18 / 36 |
| Reference recall, all promos | 8 / 17 (47 %) | **13 / 17 (76 %)** | 12 / 17 (71 %) |
| Reference recall, recommended only | 5 / 17 | 5 / 17 | 4 / 17 |
| Critic probes correct | 6 / 6* | 6 / 6 | 6 / 6 |

**Why v2 missed 9 references:** all 9 *had been proposed*, then lost: the LLM repair returned the same broken range (R02, R13), one low subjective score rejected an 81–82/100 clip (R07, R05), or a cross-episode setup was never paired (R08). **v3** fixes 27 of 30 invalid clips by trimming in code and sends one-weakness clips to review instead of rejecting them (4 live requests). Recall over all promos rose 47 % → 76 %, but every new match arrives as "needs review". Of the 19 new promos, 9 are good, 5 borderline and 5 poor, so the quality-adjusted gain is +3, not +5. **v3 is the final result:** the clean list grew (14 → 17) and weaker additions are labelled (`RECALL_STUDY.md` in the raw output).

**v4** (9 live requests) re-scouts the 15 episodes without a clip: 7 new promos (3 good, 3 borderline, 1 poor marked recommended), 40 episodes. It lost R03 to a dedup rule since fixed; replayed with the fix it equals v3's recall with 19 recommended. Arc planner v2 still missed R08, so v4 was not adopted. *Probes and the 75-point rule were set after seeing data, and the references were picked by an AI, so recall is a consistency check, not ground truth.

## 5. Assumptions
Script times match the video; cuts fall on block boundaries (shots may be trimmed, lines never); the hook beat starts within 3 s; a new viewer knows only the clip; spoilers affect order, not score.

## 6. Models, free tier and scale
One `complete(system, prompt)` interface: Claude (default), Gemini, Groq, OpenRouter, Ollama, any OpenAI-compatible API, cache-only replay, or an offline heuristic; switching is one flag. For free tiers it paces requests, retries on HTTP 429, stops on the daily quota and resumes from its cache. `--batch` puts 10 episodes or 15 clips in a request (about 16 requests instead of 250).

## 7. Why results are limited, and how to improve them
All-promo recall is 76 % (47 % was v2), but **only 5 / 17 references are among the recommended clips**: most good finds wait in the review queue. v3's own output shows why:
- **The free models are imprecise.** 30 of 82 proposed clips broke a timing rule (code fixed 27), and 7 promos carry a hook or cliffhanger description their clip does not support.
- **Their judgement is noisy.** Of 31 promos sent to review, 15 have scout and critic scores ≥ 15 points apart and 11 one weak subjective score. Batching 10 episodes or 15 clips per request spreads attention and makes reviews less independent.
- **Arcs are planned from loglines**, so long-range pay-offs (R08) are missed; the 17 AI-picked references are a small, subjective yardstick; and text cannot show acting or pace.

**A stronger model** (e.g. Claude Opus 5.5: `--provider anthropic`, no code change) targets the model errors: more exact block choices, descriptions that match the clip, steadier scores (fewer disagreement and borderline flags), and, with a paid quota, one request per episode and per clip instead of batches. It cannot fix the text-only limit or the subjective references. **Not yet measured:** run the same command with `--provider anthropic`, then `compare` it with v3. **Other levers:** references and probes labelled by editors, to set thresholds on real judgement; `--per-episode 5` for more candidates; two critics that must agree; arc planning from full scripts; a check of the rendered video before publishing.

## 8. AI tools used, and where AI got it wrong
Claude Code (Claude Opus 5.5) read the scripts and wrote the code, tests, reference picks and docs; Gemini (free tier) runs the pipeline. Mistakes the system caught, all fixed and tested:
- **The independent verifier caught an AI-written validator** letting a promo end 0.2 s into the line *"Harrington! Gerald Harrington paid us!"*.
- **The audit found five promos whose cliffhanger shot is never on screen** (the validator counted trimmed shots); P010's description invented *"as beams collapse"*.
- **Gemini wrote `"narrative_ coherence"`**, read as 0/10 for P029; keys are now normalised. **Repairs often returned the same broken range**; code trimming now runs first.
