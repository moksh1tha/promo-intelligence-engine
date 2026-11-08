# Promo Finder: solution

## 1. What it does
Reads timestamped screenplays and returns ranked 20–60 s promos (segments, hook, cliffhanger, rationale, score, publishing plan) plus every rejected candidate with the reason. It only selects moments.

## 2. How it works (five stages)
1. **Parse**: scripts become numbered blocks (dialogue, ACTION, SFX, CAMERA) with source times (3,727 blocks).
2. **Scout (LLM)**: proposes clips per episode, plus cross-episode ideas planned from the loglines. **It picks block numbers, never timestamps**; times are copied from the script, so they cannot be hallucinated. It can still misdescribe what it picked, hence stages 3–5.
3. **Validate (code)**: 20–60 s; hook starts in the first 3 s; clip ends ≤ 4 s after the cliffhanger; hook and cliffhanger supported by the selected screenplay blocks (inside the cut, not in a trimmed shot); no dialogue cut mid-line; segments in story order. A failing clip is first **repaired by trimming its edges in code**, then gets one LLM repair attempt; every change is logged.
4. **Blind critic (LLM)**: a second prompt sees only the clip, as a first-time viewer would, and scores the 100-point rubric. It flags weak hook, missing context, predictable ending, low curiosity and misleading edit, and its scores replace the scout's; clear failures are rejected. **One weak subjective score on a clip totalling 75 or more goes to human review instead of rejection.** Reviews of identical clips are reused from the cache.
5. **Rank and review**: dedups, ranks, writes a publishing plan and runs local **human-review checks**. An **independent verifier** (own parser) then re-checks every timestamp, duration and score against the raw scripts.
   - *Criteria:* critic scores for hook (25), standalone clarity (20), cliffhanger (25), emotion (15) and coherence (10), plus duration and boundary quality from code (5) = 100.
   - *Mechanism (code, not LLM):* clips without a borderline score first, then total score, stronger hook, shorter clip; of two overlapping clips the unflagged one stays. The verifier fails any output out of this order.
   - *Publishing order:* the rank (1–5 High, 6–10 Medium, rest Low). One LLM call explains each rank and writes a plan weighing hook strength, context needed, spoiler risk (early-story first) and arc variety; it cannot reorder.

## 3. Validation and the human in the loop
Free LLMs are imperfect, so the tool states its uncertainty. Each review flag names its evidence and suggests **accept / inspect / repair / reject**; subjective concerns never auto-reject, and an editor decides.

| Kind | Checks | Basis |
|---|---|---|
| Objective | hook late or outside the cut; cliffhanger outside it; camera-only hook; split sentence; story order; displayed vs source times; missing or zero critic score | exact, from blocks |
| Approximate | hook/cliffhanger description not supported by the clip | word overlap; asks for a look |
| Subjective | critic flags; low coherence; scout vs critic disagree by ≥ 15 points | LLM judgement |

`audit` re-checks completed runs, while `compare` evaluates different runs with the same method; both make **no model calls**. Six known good/bad clips check the critic (v3 reuses earlier scores for them). 62 tests cover broken clips (weak hooks, misleading endings, missing context, mid-line cuts), tampered outputs, quota errors and cache accounting.

## 4. Results and the recall study
**The raw output holds v3, the final run.** v2 (first version) and v4 (an experiment) were development runs, shown only for comparison: same models (free Gemini, batch mode) and match rule (a promo covers ≥ 50 % of a reference).

| | v2 (first version) | **v3 (final, in raw output)** | v4 (experiment) |
|---|---|---|---|
| Promos: recommended / needs review | 31: 14 / 17 | 48: **17 / 31** | 54: 18 / 36 |
| Reference recall, all promos | 8 / 17 (47 %) | **13 / 17 (76 %)** | 12 / 17 (71 %) |
| Reference recall, recommended only | 5 / 17 | 5 / 17 | 4 / 17 |
| Critic probes correct | 6 / 6* | 6 / 6 | 6 / 6 |

**Why v2 missed 9 references:** all 9 *had been proposed*, then lost: the LLM repair returned the same broken range (R02, R13), one low subjective score rejected an 81–82/100 clip (R07, R05), or a cross-episode setup was never paired (R08). **v3** fixes 27 of 30 invalid clips by trimming in code and sends one-weakness clips to review instead of rejecting them. Recall over all promos rose 47 % → 76 %: of the 13 matched references, 5 are among recommended promos and 8 are flagged for review, so **76 % is discovery recall, not publication-ready recall**. Of 19 new promos, 9 are good, 5 borderline, 5 poor (quality-adjusted gain +3, not +5). **v3 is the final result:** the clean list grew (14 → 17) and weaker additions are labelled (`RECALL_STUDY.md` in the raw output).

**v4** (re-scouting episodes without a clip, arc planner v2) added 7 promos and 5 episodes but no recall (with a dedup fix it equals v3) and missed R08 again, so it was not adopted. *Probes and the 75-point rule were set after seeing data, and the references were picked by an AI, so recall is a consistency check, not ground truth.

## 5. Assumptions, models and scale
**Assumptions:** script times match the video; cuts fall on block boundaries (shots may be trimmed, lines never); a new viewer knows only the clip; spoilers affect order, not score. **Models:** one `complete(system, prompt)` interface: Claude (default), Gemini, Groq, OpenRouter, Ollama, any OpenAI-compatible API, cache-only replay, or an offline heuristic; switching is one flag. For free tiers it paces requests, retries on HTTP 429, stops on the daily quota and resumes from its cache. `--batch` puts 10 episodes or 15 clips in a request (about 16 requests instead of 250).

## 6. Why results are limited, and how to improve them
Discovery recall is 76 %, but **only 5 / 17 references are among the recommended clips**. v3's own output shows why:
- **The free models are imprecise.** 30 of 82 proposed clips broke a timing rule (code fixed 27), and 7 promos carry a hook or cliffhanger description their clip does not support.
- **Their judgement is noisy.** Of 31 promos sent to review, 15 have scout and critic scores ≥ 15 points apart and 11 one weak subjective score; 5 have both, so 21 carry either flag and 10 were flagged for other reasons. Batching 10 episodes or 15 clips per request also makes reviews less independent.
- **Arcs are planned from loglines**, so long-range pay-offs (R08) are missed; the 17 AI-picked references are a small, subjective yardstick; and text cannot show acting or pace.

**A stronger model** (e.g. Claude Opus 5.5, `--provider anthropic`, no code change) targets the model errors: more exact block choices, accurate descriptions, steadier scores and, with a paid quota, unbatched requests. It cannot fix the text-only limit or the subjective references; **not yet measured** (run it, then `compare` with v3). **Other levers:** editor-labelled references and probes to set thresholds; `--per-episode 5`; two critics that must agree; arc planning from full scripts; a rendered-video check.

## 7. AI tools used, and where AI got it wrong
Claude Code (Claude Opus 5.5) wrote the code, tests, reference picks and docs; free Gemini runs the pipeline. Mistakes caught, all fixed and tested: **the independent verifier caught an AI-written validator** letting a promo end 0.2 s into a line of dialogue; **the audit found five promos whose cliffhanger block fell outside the cut** (the validator counted trimmed shots); and **Gemini wrote `"narrative_ coherence"`**, read as 0/10 for P029, so keys are now normalised.
