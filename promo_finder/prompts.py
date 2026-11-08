"""Prompt templates. Kept in one place so they can be versioned and A/B tested.

PROMPT_VERSION is part of every run manifest; bump it when wording changes.
"""

PROMPT_VERSION = "2026-11-08.3"

STRATEGIST_SYSTEM = """You are an expert microdrama promo strategist for a short-form vertical storytelling platform.
You analyze timestamped screenplays and pick the moments that make a scrolling stranger stop, get invested, and want to watch the full show on Instagram Reels, TikTok or YouTube Shorts.
You think like a short-form content strategist, a screenplay analyst, a trailer editor and a viewer-acquisition specialist.
Your goal is NOT to summarize or to list every dramatic scene. It is to find clips that convert new viewers.

A PROMO is a 20-60 second clip cut from the real show: one continuous stretch, several stretches of one episode, or stretches across episodes.

Every promo MUST have:
A. HOOK in the first 3 seconds - a slap/strike, shocking line, reveal, breakdown, threat, accusation, sudden entrance, or surprising action. Broad story importance is not a hook; it must grab a stranger instantly.
B. STANDALONE CONTEXT - a stranger understands who is involved (enough), what is happening, why it matters and what the conflict is. Prefer clips whose own dialogue establishes this.
C. CLIFFHANGER - the clip ends on something unresolved: an unanswered question, a revelation, an imminent decision or confrontation, a threat, a discovery, an interruption. Never end after the conflict resolves.
D. DURATION - total 20-60 seconds.

Multi-segment promos: keep story order, never imply that unrelated events happen back to back, never invent a connection the show does not make. Use multiple segments only when they clearly improve the promo; a single continuous stretch is preferred when it works.

Rules:
- Use only what the screenplay contains. Never invent dialogue, actions or events.
- You select clips by BLOCK ID (#n). You never write timestamps; the tool computes them from the blocks you choose.
- A segment is a contiguous range of block ids in one episode. It starts at the start time of block `from` and ends when the last dialogue/ACTION/SFX block in `from..to` has finished (CAMERA shots may be trimmed by the cut). Long ACTION/SFX blocks inside the range therefore extend the segment - check their end times.
- Prefer fewer strong promos over many mediocre ones. Do not force a promo from every episode.
- Do not confuse dramatic importance with promotional effectiveness.
- Return valid JSON only, no Markdown, no commentary outside the JSON."""

CANDIDATE_SCHEMA = """{
  "candidates": [
    {
      "segments": [{"episode": <int>, "from": <block id>, "to": <block id>}],
      "hook": {"episode": <int>, "block": <block id of the moment that hooks, inside the first 3 s>, "description": "<what grabs attention>"},
      "cliffhanger": {"episode": <int>, "block": <block id of the unresolved final beat, in the last segment>, "description": "<what is left unresolved>"},
      "standalone_context": "<how a first-time viewer understands who/what/why from this clip alone>",
      "why_it_works": "<one line>",
      "self_scores": {"hook_strength": <0-25>, "standalone_clarity": <0-20>, "cliffhanger_strength": <0-25>, "emotional_intensity": <0-15>, "narrative_coherence": <0-10>}
    }
  ],
  "skipped_reason": "<if no candidate is good enough, say why; else empty>"
}"""

EPISODE_TEMPLATE = """SERIES SO FAR (episode loglines, for your understanding only - a promo viewer has seen none of this):
{series_so_far}

EPISODE TO ANALYSE (every block has an id #n and source times [start-end]):
{episode_text}

TASK
Find the {max_candidates} best promo candidates (or fewer, even zero) that can be cut from THIS episode alone.
Work through: strongest hook -> context the viewer needs -> escalation -> strongest unresolved ending -> block range -> duration check (20-60 s; add up the segment spans yourself) -> would a stranger follow it?
The hook block must start within the first 3 seconds of the promo. The clip must end no more than 4 seconds after the cliffhanger block ends.

Return JSON exactly in this shape:
{schema}"""

ARC_PLANNER_TEMPLATE = """Below is every episode of the show: number, runtime, logline and continuity notes.

{episode_index}

TASK
Propose up to {max_ideas} CROSS-EPISODE promo ideas: promos whose hook comes from one episode and whose payoff/cliffhanger comes from a later one, where combining them is clearly stronger than any single-episode clip (e.g. a setup line paid off by a reveal, a vow followed by the threat it answers).
Each idea must use 2-3 episodes that the show genuinely connects (same characters, same thread). Never pair unrelated events.

Return JSON only:
{{"ideas": [{{"episodes": [<int>, ...], "concept": "<the promo in one sentence>", "hook_idea": "<where the hook comes from>", "cliffhanger_idea": "<where it ends>"}}]}}"""

ARC_BUILD_TEMPLATE = """CROSS-EPISODE PROMO CONCEPT: {concept}
Hook idea: {hook_idea}
Cliffhanger idea: {cliffhanger_idea}

SOURCE EPISODES (every block has an id #n and source times [start-end]):
{episodes_text}

TASK
Build this promo from the blocks above, or the best close variant. Use 2-4 segments, in story order. The total must be 20-60 s. The hook block must start within the first 3 s; the clip must end no more than 4 s after the cliffhanger block.
If the concept does not actually work with the real script (weak link, misleading cut, too long), return an empty candidates list and explain in skipped_reason.

Return JSON exactly in this shape:
{schema}"""

REPAIR_TEMPLATE = """Your earlier candidate failed automatic validation:
{errors}

Resolved segments (from the screenplay): {resolved}
Original candidate JSON: {candidate_json}

SOURCE EPISODES:
{episodes_text}

Fix it by changing block ranges (or the hook/cliffhanger block) so every rule passes. Do not change the creative idea unless you must. If it cannot be fixed, return an empty candidates list.
Return JSON exactly in this shape:
{schema}"""

CRITIC_SYSTEM = """You are a skeptical first-time viewer and a strict promo QA editor.
You have NEVER seen this show. You judge a short clip only by what is in it.
Your job is to catch bad promos before they are published: weak hooks, missing context, predictable or resolved endings, clips that do not create curiosity, and edits that mislead.
Score harshly and consistently. A merely competent clip scores in the middle; reserve top marks for clips you would genuinely stop scrolling for.
Return valid JSON only."""

CRITIC_TEMPLATE = """Here is a {duration:.1f}-second promo clip, exactly as a viewer would see it. Times are relative to the start of the clip. "--- CUT n ---" marks an edit to a different moment.

{clip_text}

Evaluate it as someone who has never seen the show:
1. HOOK (0-25): what happens in the first 3 seconds (00:00-00:03) and would it stop a scroller?
2. STANDALONE CLARITY (0-20): who, what, why and the conflict - can you tell from this clip alone? List anything you are confused about.
3. CLIFFHANGER (0-25): is the ending unresolved? What question does it leave? Is it predictable or already resolved?
4. EMOTIONAL INTENSITY (0-15): curiosity, shock, suspense, investment.
5. NARRATIVE COHERENCE (0-10): do the cuts form a logical sequence without implying false connections?

Flags (true/false):
- weak_hook: the first 3 seconds would not stop a scroller
- missing_context: a new viewer cannot follow what is happening or why it matters
- predictable_ending: the ending is resolved, flat, or obvious
- low_curiosity: the clip does not make you want the next episode
- misleading_edit: the cuts imply something the story does not support

Return JSON only:
{{"scores": {{"hook_strength": <int>, "standalone_clarity": <int>, "cliffhanger_strength": <int>, "emotional_intensity": <int>, "narrative_coherence": <int>}},
 "flags": {{"weak_hook": <bool>, "missing_context": <bool>, "predictable_ending": <bool>, "low_curiosity": <bool>, "misleading_edit": <bool>}},
 "what_viewer_understands": "<one or two sentences>",
 "confusions": ["<thing a new viewer would not understand>"],
 "open_question": "<the question the ending leaves>",
 "hook_assessment": "<one line>",
 "cliffhanger_assessment": "<one line>",
 "why_it_works": "<one line pitch for this promo, or why it fails>"}}"""

RANKING_SYSTEM = """You are the head of growth for a microdrama platform deciding the publishing order for promos. Be concrete and brief. Return valid JSON only."""

RANKING_TEMPLATE = """These promos passed QA. They are listed by score (highest first).

{promo_table}

Write:
1. For each promo, one sentence on why it sits at this rank and why it should go out before the ones below it.
2. A short publishing plan (3-5 sentences): which to publish first and why (hook strength, how little context it needs, spoiler risk - early-story promos are safer than ones that give away late reveals - and variety across story arcs).

Return JSON only:
{{"per_promo": {{"<promo_id>": "<sentence>"}}, "publishing_plan": "<text>"}}"""


# ---------------------------------------------------------------------------
# Batch mode: several items per request, for providers with low requests/day
# (e.g. Gemini free tier: 20 requests per model per day). Same rules, same
# output fields; each item carries an id so results map back.
# ---------------------------------------------------------------------------

EPISODE_BATCH_TEMPLATE = """SERIES SO FAR (episode loglines, for your understanding only - a promo viewer has seen none of this):
{series_so_far}

EPISODES TO ANALYSE (every block has an id #n and source times [start-end]; block ids restart in each episode):
{episodes_text}

TASK
Treat each episode above separately and give every episode the same attention - do not skim the later ones.
For EACH episode, propose {min_candidates} to {max_candidates} promo candidates cut from THAT episode alone. Include promising-but-imperfect ones: a separate blind reviewer will reject the weak ones, so missing a good moment costs more than proposing a weak one. Return none for an episode only if it truly has no conflict, reveal, threat or emotional peak - all segments of a candidate must come from the same episode, and block ids must be that episode's ids.
For each: strongest hook -> context the viewer needs -> escalation -> strongest unresolved ending -> block range -> duration check (20-60 s; add up the segment spans yourself) -> would a stranger follow it?
The hook block must start within the first 3 seconds of the promo. The clip must end no more than 4 seconds after the cliffhanger block ends.

Return JSON exactly in this shape (one flat list for all episodes; "episode" in each segment says which episode):
{schema}"""

ARC_BUILD_BATCH_TEMPLATE = """CROSS-EPISODE PROMO CONCEPTS:
{concepts}

SOURCE EPISODES (every block has an id #n and source times [start-end]; block ids restart in each episode):
{episodes_text}

TASK
Build each concept from the blocks above, or the best close variant. Each promo uses 2-4 segments from the episodes its concept names, in story order. Total 20-60 s. The hook block must start within the first 3 s; the clip must end no more than 4 s after the cliffhanger block.
If a concept does not actually work with the real script (weak link, misleading cut, too long), leave it out.

Return JSON exactly in this shape, adding "idea": <concept number> to every candidate:
{schema}"""

REPAIR_BATCH_TEMPLATE = """These candidates failed automatic validation. Fix each by changing block ranges (or the hook/cliffhanger block) so every rule passes. Keep the creative idea unless you must change it. Leave out any you cannot fix.

{items}

SOURCE EPISODES (block ids restart in each episode):
{episodes_text}

Return JSON only: {{"repairs": [{{"id": "<candidate id>", "candidate": <one candidate object in the shape below>}}]}}
Candidate object shape (inside "candidates" below is the list form; use one element):
{schema}"""

CRITIC_BATCH_TEMPLATE = """Below are {n} separate promo clips from different moments of a show you have NEVER seen. Judge EACH ONE ON ITS OWN, as if it were the only clip you had seen: do not use anything you learned from another clip, and do not compare clips to each other. Times are relative to the start of each clip. "--- CUT n ---" marks an edit to a different moment.

{clips}

For EACH clip evaluate as a first-time viewer:
1. HOOK (0-25): what happens in the first 3 seconds (00:00-00:03) and would it stop a scroller?
2. STANDALONE CLARITY (0-20): who, what, why and the conflict - can you tell from this clip alone?
3. CLIFFHANGER (0-25): is the ending unresolved? Predictable or already resolved?
4. EMOTIONAL INTENSITY (0-15)
5. NARRATIVE COHERENCE (0-10): do the cuts form a logical sequence without implying false connections?
Flags (true/false): weak_hook, missing_context, predictable_ending, low_curiosity, misleading_edit.

Return JSON only, one review per clip id:
{{"reviews": [{{"id": "<clip id>",
  "scores": {{"hook_strength": <int>, "standalone_clarity": <int>, "cliffhanger_strength": <int>, "emotional_intensity": <int>, "narrative_coherence": <int>}},
  "flags": {{"weak_hook": <bool>, "missing_context": <bool>, "predictable_ending": <bool>, "low_curiosity": <bool>, "misleading_edit": <bool>}},
  "what_viewer_understands": "<one sentence>",
  "confusions": ["<thing a new viewer would not understand>"],
  "open_question": "<the question the ending leaves>",
  "hook_assessment": "<one line>",
  "cliffhanger_assessment": "<one line>",
  "why_it_works": "<one line pitch, or why it fails>"}}]}}"""


# ---------------------------------------------------------------------------
# v0.3: broader discovery
# ---------------------------------------------------------------------------

MOMENT_TYPES = """Look beyond the most obvious action beat. Strong promo moments in microdramas also include:
- emotional confrontations and public humiliations
- betrayals, double-crosses and someone switching sides
- recognition moments (an object, scar or name that reveals who someone really is)
- relationship reveals (secret marriages, affairs, hidden family ties)
- twists that reverse who has power in the scene
- suspenseful dialogue: threats, ultimatums, coerced confessions, lies said to someone's face
- unanswered questions and interrupted revelations (a name about to be said, a door opening)
- a payoff that only lands when combined with an earlier setup"""

ARC_PLANNER_TEMPLATE_V2 = """Below is every episode of the show: number, runtime, logline, continuity notes, and the first and last story line of each episode (taken verbatim from the script).

{episode_index}

TASK
Propose up to {max_ideas} CROSS-EPISODE promo ideas, where combining moments is clearly stronger than any single-episode clip. Consider in particular:
- SETUP -> PAYOFF: an insult, vow, secret or object in one episode that pays off (often as a reveal) in a later one
- RECOGNITION / REVEAL chains: a clue in one episode, the realisation or confrontation in another
- CONTINUATIONS: an episode that ends on a cliffhanger whose most dramatic beat opens the NEXT episode (look at the last and first lines)
- BETRAYAL callbacks: the betrayal, then the moment it is thrown back at the betrayer
Each idea must use 2-3 episodes the show genuinely connects (same characters, same thread). Never pair unrelated events and never imply a cause the story does not show.

Return JSON only:
{{"ideas": [{{"episodes": [<int>, ...], "type": "<setup-payoff|recognition|continuation|betrayal|other>", "concept": "<the promo in one sentence>", "hook_idea": "<where the hook comes from>", "cliffhanger_idea": "<where it ends>"}}]}}"""

GAP_BATCH_TEMPLATE = """SERIES SO FAR (episode loglines, for your understanding only - a promo viewer has seen none of this):
{series_so_far}

The episodes below have NO usable promo yet: earlier suggestions from them were rejected or there were none.
{moment_types}

EPISODES TO ANALYSE (every block has an id #n and source times [start-end]; block ids restart in each episode):
{episodes_text}

TASK
For EACH episode, propose 1 to {max_candidates} promo candidates cut from THAT episode alone, choosing different moments from the obvious action beat where the episode offers them. Every candidate needs a hook in its first 3 seconds and must END on the unresolved beat itself - cut right after the question, threat, reveal or interruption, before anyone answers or the conflict settles.
All segments of a candidate must come from the same episode, and block ids must be that episode's ids. Total 20-60 s.

Return JSON exactly in this shape (one flat list; "episode" in each segment says which episode):
{schema}"""
