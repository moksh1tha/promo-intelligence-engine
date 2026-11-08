"""Run with:  python3 -m unittest discover -s tests -v"""
import json
import os
import shutil
import tempfile
import unittest
import unittest.mock

from promo_finder import timecode
from promo_finder.candidates import BlockRef, Candidate, Segment, render_clip, validate
from promo_finder.critic import apply_critique
from promo_finder.fixtures import candidate_from_times, segment_from_times
from promo_finder.llm import LLM, extract_json
from promo_finder.pipeline import Run
from promo_finder.screenplay import load_show
from promo_finder.verify import check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
if not os.path.isdir(DATA) or not [f for f in os.listdir(DATA) if f.endswith(".txt")]:
    # The input pack is Kuku's confidential content and is not in the public repository.
    raise unittest.SkipTest("input pack not found: put the 50 episode .txt files in data/ to run the tests")
SHOW = load_show(DATA)


def cand(segs, hook=None, cliff=None, cid="t"):
    segments = [Segment(e, a, b) for e, a, b in segs]
    hook = hook or (segs[0][0], segs[0][1])
    cliff = cliff or (segs[-1][0], segs[-1][2])
    return validate(SHOW, Candidate(cid, "test", segments, BlockRef(*hook), "", BlockRef(*cliff), "", ""))


class TestParsing(unittest.TestCase):
    def test_full_pack_parsed(self):
        self.assertEqual(sorted(SHOW), list(range(1, 51)))
        raw = 0
        for f in os.listdir(DATA):
            with open(os.path.join(DATA, f), encoding="utf-8") as fh:
                raw += fh.read().count("Start: (")
        self.assertEqual(sum(len(e.blocks) for e in SHOW.values()), raw, "every source block must be parsed")

    def test_blocks_sorted_and_ids_stable(self):
        for ep in SHOW.values():
            self.assertEqual([b.id for b in ep.blocks], list(range(len(ep.blocks))))
            self.assertEqual([b.start for b in ep.blocks], sorted(b.start for b in ep.blocks))

    def test_known_line(self):
        ep1 = SHOW[1]
        line = next(b for b in ep1.blocks if "You should turn around" in b.text)
        self.assertEqual((line.start, line.end, line.kind, line.speaker), (11.3, 13.1, "dialogue", "MATRIARCH"))

    def test_timecode_roundtrip(self):
        self.assertEqual(timecode.parse("01:30.20"), 90.2)
        self.assertEqual(timecode.parse("00:01:30.20"), 90.2)
        self.assertEqual(timecode.fmt(90.2), "00:01:30.20")


class TestValidationCatchesBadSelections(unittest.TestCase):
    """Deterministic QC: incorrect timestamps, durations, hook/cliffhanger placement."""

    def test_valid_reference_clip(self):
        c = candidate_from_times(SHOW, "ok", [(3, "00:58.30", "01:25.50"), (3, "01:59.80", "02:25.75")])
        self.assertEqual(c.errors, [])
        self.assertAlmostEqual(c.total_duration, 53.15)

    def test_nonexistent_block_ids(self):
        c = cand([(3, 10, 9999)])
        self.assertTrue(any("outside episode" in e for e in c.errors))

    def test_nonexistent_episode(self):
        c = cand([(99, 0, 5)])
        self.assertTrue(any("does not exist" in e for e in c.errors))

    def test_timestamp_not_on_a_block_boundary(self):
        with self.assertRaises(ValueError):
            segment_from_times(SHOW, 3, "00:58.31", "01:25.50")

    def test_too_short(self):
        s = segment_from_times(SHOW, 1, "00:11.30", "00:19.40")
        c = cand([(1, s.from_block, s.to_block)])
        self.assertTrue(any("below 20" in e for e in c.errors))

    def test_too_long(self):
        s = segment_from_times(SHOW, 1, "00:11.30", "01:30.80")
        c = cand([(1, s.from_block, s.to_block)])
        self.assertTrue(any("exceeds 60" in e for e in c.errors))

    def test_hook_outside_first_3_seconds(self):
        s = segment_from_times(SHOW, 1, "00:11.30", "00:56.90")
        late = next(b.id for b in SHOW[1].blocks if b.start >= 30)
        c = cand([(1, s.from_block, s.to_block)], hook=(1, late))
        self.assertTrue(any("outside the first 3s" in e for e in c.errors))

    def test_clip_continues_after_cliffhanger(self):
        s = segment_from_times(SHOW, 1, "00:11.30", "00:56.90")
        early = next(b.id for b in SHOW[1].blocks if "Not anymore" in b.text)
        c = cand([(1, s.from_block, s.to_block)], cliff=(1, early))
        self.assertTrue(any("past the cliffhanger" in e for e in c.errors))

    def test_segments_out_of_order(self):
        a = segment_from_times(SHOW, 3, "01:59.80", "02:25.75")
        b = segment_from_times(SHOW, 3, "00:58.30", "01:25.50")
        c = cand([(3, a.from_block, a.to_block), (3, b.from_block, b.to_block)], hook=(3, a.from_block),
                 cliff=(3, b.to_block))
        self.assertTrue(any("chronological" in e for e in c.errors))

    def test_cut_mid_dialogue(self):
        # Ep16: ACTION #40 runs 48.0-50.0 while the line "Harrington! Gerald Harrington paid us!" (#43)
        # starts at 49.8. A range ending at #42 therefore ends at 50.0, inside that line.
        ep = SHOW[16]
        self.assertIn("Gerald Harrington paid us", ep.blocks[43].text)
        start = next(b.id for b in ep.blocks if b.kind == "dialogue" and b.start >= 20)
        c = cand([(16, start, 42)], cliff=(16, 41))
        self.assertTrue(any("ends mid-line" in e for e in c.errors), c.errors)


class TestCriticDecision(unittest.TestCase):
    def _ok(self):
        return candidate_from_times(SHOW, "ok", [(3, "00:58.30", "01:25.50"), (3, "01:59.80", "02:25.75")])

    def test_flag_with_weak_matching_score_rejects(self):
        weak = {"weak_hook": "hook_strength", "missing_context": "standalone_clarity",
                "predictable_ending": "cliffhanger_strength", "low_curiosity": "emotional_intensity",
                "misleading_edit": "narrative_coherence"}
        for flag, score_key in weak.items():
            scores = {"hook_strength": 22, "standalone_clarity": 18, "cliffhanger_strength": 22,
                      "emotional_intensity": 13, "narrative_coherence": 9}
            scores[score_key] = 2
            c = apply_critique(self._ok(), {"scores": scores, "flags": {flag: True}}, policy="strict")
            self.assertTrue(any(r.startswith("critic:") for r in c.rejection_reasons), flag)

    def test_flag_with_good_matching_score_is_only_a_warning(self):
        for flag in ("weak_hook", "missing_context", "predictable_ending", "low_curiosity", "misleading_edit"):
            c = apply_critique(self._ok(), {"scores": {"hook_strength": 22, "standalone_clarity": 18,
                                                       "cliffhanger_strength": 22, "emotional_intensity": 13,
                                                       "narrative_coherence": 9}, "flags": {flag: True}})
            self.assertEqual(c.rejection_reasons, [], flag)
            self.assertTrue(c.critique["warnings"], flag)

    def test_low_scores_reject_without_flags(self):
        c = apply_critique(self._ok(), {"scores": {"hook_strength": 8, "standalone_clarity": 18,
                                                   "cliffhanger_strength": 22, "emotional_intensity": 13,
                                                   "narrative_coherence": 9}, "flags": {}})
        self.assertTrue(any("hook score" in r for r in c.rejection_reasons))

    def test_scores_clamped_and_summed(self):
        c = apply_critique(self._ok(), {"scores": {"hook_strength": 99, "standalone_clarity": "18",
                                                   "cliffhanger_strength": 22, "emotional_intensity": 13,
                                                   "narrative_coherence": 9}, "flags": {}})
        self.assertEqual(c.scores["hook_strength"], 25)
        self.assertEqual(c.total_score, sum(c.scores.values()))
        self.assertEqual(c.rejection_reasons, [])


class TestJsonExtraction(unittest.TestCase):
    def test_fenced(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_prose_around(self):
        self.assertEqual(extract_json('Here you go: {"a": [1, 2]} thanks'), {"a": [1, 2]})


class FakeLLM(LLM):
    """Deterministic stand-in for a model: proves the LLM path works end to end offline."""
    provider = "fake"

    def _call(self, system, prompt):
        if "CROSS-EPISODE promo ideas" in prompt:
            return json.dumps({"ideas": [{"episodes": [26, 27], "concept": "ring recognised", "hook_idea": "gasp",
                                          "cliffhanger_idea": "doors burst"}]})
        if "CROSS-EPISODE PROMO CONCEPT" in prompt:
            ref = candidate_from_times(SHOW, "x", [(26, "01:30.20", "01:55.50"), (27, "00:17.20", "00:31.40"),
                                                   (27, "00:44.50", "00:51.50")])
            return json.dumps({"candidates": [{
                "segments": [{"episode": s.episode, "from": s.from_block, "to": s.to_block} for s in ref.segments],
                "hook": {"episode": 26, "block": ref.hook.block, "description": "gasp"},
                "cliffhanger": {"episode": 27, "block": ref.cliffhanger.block, "description": "doors"},
                "why_it_works": "x", "self_scores": {"hook_strength": 20}}]})
        if "failed automatic validation" in prompt:
            s = segment_from_times(SHOW, 3, "00:58.30", "01:25.50")
            return json.dumps({"candidates": [{"segments": [{"episode": 3, "from": s.from_block, "to": s.to_block}],
                                               "hook": {"episode": 3, "block": s.from_block},
                                               "cliffhanger": {"episode": 3, "block": s.to_block}}]})
        if "EPISODE TO ANALYSE" in prompt:
            if "EPISODE 3 " in prompt:   # deliberately too long -> must be repaired
                return json.dumps({"candidates": [{"segments": [{"episode": 3, "from": 0, "to": len(SHOW[3].blocks) - 1}],
                                                   "hook": {"episode": 3, "block": 0},
                                                   "cliffhanger": {"episode": 3, "block": len(SHOW[3].blocks) - 1}}]})
            return json.dumps({"candidates": [], "skipped_reason": "nothing strong"})
        if "promo clip" in prompt:
            return json.dumps({"scores": {"hook_strength": 21, "standalone_clarity": 16, "cliffhanger_strength": 21,
                                          "emotional_intensity": 12, "narrative_coherence": 8},
                               "flags": {}, "why_it_works": "fake"})
        if "passed QA" in prompt:
            return json.dumps({"per_promo": {}, "publishing_plan": "fake plan"})
        raise AssertionError("unexpected prompt")


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_llm_path_with_repair_and_cross_episode(self):
        out = os.path.join(self.tmp, "run")
        run = Run(DATA, out, FakeLLM("fake-model", cache_dir=os.path.join(self.tmp, "cache")),
                  episodes=[3, 26, 27], workers=2, reference=None, probes=None, log=lambda m: None)
        doc = run.execute()
        self.assertEqual(doc["analysis_summary"]["valid_promos"], 2)
        origins = {p["origin"] for p in doc["ranked_promos"]}
        self.assertIn("arc01", origins)
        self.assertTrue(any(r["stage_rejected"] == "validation" for r in doc["rejected_candidates"]))
        errors, _ = check(DATA, doc)
        self.assertEqual(errors, [])
        self.assertTrue(os.path.exists(os.path.join(out, "raw", "scout_ep003.json")))
        self.assertTrue(os.path.exists(os.path.join(out, "report.md")))

    def test_heuristic_full_pack_verifies(self):
        out = os.path.join(self.tmp, "h")
        doc = Run(DATA, out, None, reference=None, probes=None, log=lambda m: None).execute()
        self.assertEqual(len(doc["analysis_summary"]["episodes_analyzed"]), 50)
        self.assertGreater(doc["analysis_summary"]["valid_promos"], 0)
        errors, _ = check(DATA, doc)
        self.assertEqual(errors, [])


class TestVerifierCatchesTampering(unittest.TestCase):
    def test_tampered_timestamp(self):
        doc = Run(DATA, tempfile.mkdtemp(), None, episodes=[1, 2, 3], reference=None, probes=None,
                  log=lambda m: None).execute()
        p = doc["ranked_promos"][0]
        p["segments"][0]["start_timestamp"] = "00:00:00.37"
        errors, _ = check(DATA, doc)
        self.assertTrue(any("not a block start" in e for e in errors))


if __name__ == "__main__":
    unittest.main()


class TestAnthropicAdapter(unittest.TestCase):
    """Checks request shape and refusal handling with a stub SDK (no network)."""

    def test_request_shape_and_refusal(self):
        import sys
        import types
        from promo_finder.llm import AnthropicLLM, LLMError
        sent = {}

        class Resp:
            def __init__(self, stop):
                self.stop_reason = stop
                self.content = [types.SimpleNamespace(type="text", text='{"ok": 1}')]
                self.usage = types.SimpleNamespace(input_tokens=10, output_tokens=5)

        class Messages:
            stop = "end_turn"

            def create(self, **kw):
                sent.update(kw)
                return Resp(Messages.stop)

        stub = types.ModuleType("anthropic")
        stub.Anthropic = lambda: types.SimpleNamespace(beta=types.SimpleNamespace(messages=Messages()))
        stub.APIConnectionError = type("APIConnectionError", (Exception,), {})
        old = sys.modules.get("anthropic")
        sys.modules["anthropic"] = stub
        try:
            llm = AnthropicLLM("claude-opus-5-5", effort="high")
            self.assertEqual(llm.complete("sys", "hi"), '{"ok": 1}')
            self.assertEqual(sent["model"], "claude-opus-5-5")
            self.assertEqual(sent["extra_body"]["output_config"], {"effort": "high"})
            self.assertEqual(sent["extra_body"]["fallbacks"], "default")
            self.assertIn("server-side-fallback-2026-07-01", sent["betas"])
            self.assertEqual((llm.input_tokens, llm.output_tokens), (10, 5))
            Messages.stop = "refusal"
            with self.assertRaises(LLMError):
                llm.complete("sys", "different prompt")
        finally:
            if old is None:
                del sys.modules["anthropic"]
            else:
                sys.modules["anthropic"] = old


class TestOpenAICompatibleClient(unittest.TestCase):
    """Free-tier client against a local fake server: success, 429 + Retry-After, daily quota."""

    def setUp(self):
        import http.server
        import threading
        state = self.state = {"mode": "ok", "hits": 0, "auth": None}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj, headers=None):
                body = json.dumps(obj).encode()
                self.send_response(code)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send(200, {"data": [{"id": "m-b"}, {"id": "m-a"}]})

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                state["hits"] += 1
                state["auth"] = self.headers.get("Authorization")
                if state["mode"] == "429once" and state["hits"] == 1:
                    return self._send(429, {"error": {"message": "slow down"}}, {"Retry-After": "0"})
                if state["mode"] == "daily":
                    return self._send(429, {"error": {"message": "Quota exceeded for metric: RequestsPerDay"}})
                self._send(200, {"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

        self.server = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def tearDown(self):
        self.server.shutdown()

    def _llm(self):
        from promo_finder.llm import OpenAICompatibleLLM
        return OpenAICompatibleLLM("m", self.url, "test-key", provider="gemini")

    def test_success_and_auth(self):
        llm = self._llm()
        self.assertEqual(extract_json(llm.complete("s", "p")), {"ok": True})
        self.assertEqual(self.state["auth"], "Bearer test-key")
        self.assertEqual((llm.input_tokens, llm.output_tokens), (7, 3))
        self.assertEqual(llm.list_models(), ["m-a", "m-b"])

    def test_retries_after_429(self):
        self.state["mode"] = "429once"
        self.assertEqual(extract_json(self._llm().complete("s", "p")), {"ok": True})
        self.assertEqual(self.state["hits"], 2)

    def test_daily_quota_stops_early(self):
        from promo_finder.llm import QuotaExhausted
        self.state["mode"] = "daily"
        llm = self._llm()
        with self.assertRaises(QuotaExhausted):
            llm.complete("s", "p1")
        with self.assertRaises(QuotaExhausted):
            llm.complete("s", "p2")
        self.assertEqual(self.state["hits"], 1, "after a daily-quota error no further requests are sent")

    def test_presets_need_key(self):
        from promo_finder.llm import LLMError, make_llm
        old = {k: os.environ.pop(k, None) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        try:
            with self.assertRaises(LLMError):
                make_llm("gemini", None, None)
        finally:
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v


class BatchFakeLLM(LLM):
    """Answers batch prompts: one candidate per episode (the episode-final 30 s), reviews for every clip."""
    provider = "fake-batch"

    def _call(self, system, prompt):
        import re as _re
        if "EPISODES TO ANALYSE" in prompt:
            eps = [int(x) for x in _re.findall(r"^EPISODE (\d+)  \(runtime", prompt, _re.M)]
            out = []
            for n in eps:
                blocks = SHOW[n].blocks
                story = [b for b in blocks if b.kind != "camera"]
                last = story[-1]
                start = next(b for b in story if b.start >= last.end - 35 and b.kind == "dialogue")
                out.append({"segments": [{"episode": n, "from": start.id, "to": last.id}],
                            "hook": {"episode": n, "block": start.id}, "cliffhanger": {"episode": n, "block": last.id}})
            # one deliberately broken candidate to exercise batched repair
            out.append({"segments": [{"episode": eps[0], "from": 0, "to": len(SHOW[eps[0]].blocks) - 1}],
                        "hook": {"episode": eps[0], "block": 0},
                        "cliffhanger": {"episode": eps[0], "block": len(SHOW[eps[0]].blocks) - 1}})
            return json.dumps({"candidates": out})
        if "CROSS-EPISODE promo ideas" in prompt:
            return json.dumps({"ideas": [{"episodes": [26, 27], "concept": "ring", "hook_idea": "", "cliffhanger_idea": ""}]})
        if "CROSS-EPISODE PROMO CONCEPTS" in prompt:
            ref = candidate_from_times(SHOW, "x", [(26, "01:30.20", "01:55.50"), (27, "00:17.20", "00:31.40"),
                                                   (27, "00:44.50", "00:51.50")])
            return json.dumps({"candidates": [{"idea": 1, "segments": [
                {"episode": s.episode, "from": s.from_block, "to": s.to_block} for s in ref.segments],
                "hook": {"episode": 26, "block": ref.hook.block}, "cliffhanger": {"episode": 27, "block": ref.cliffhanger.block}}]})
        if "have NO usable promo yet" in prompt:
            eps = [int(x) for x in _re.findall(r"^EPISODE (\d+)  \(runtime", prompt, _re.M)]
            out = []
            for n in eps:
                story = [b for b in SHOW[n].blocks if b.kind != "camera"]
                last = story[-1]
                start = next(b for b in story if b.start >= last.end - 28 and b.kind == "dialogue")
                out.append({"segments": [{"episode": n, "from": start.id, "to": last.id}],
                            "hook": {"episode": n, "block": start.id}, "cliffhanger": {"episode": n, "block": last.id}})
            return json.dumps({"candidates": out})
        if '"repairs"' in prompt:
            # answer in the wrapped shape real models produced ({"candidate": {"candidates": [...]}})
            out = []
            for cid in _re.findall(r"^ID (\S+)", prompt, _re.M):
                n = int(_re.search(r"(\d+)#", cid).group(1))
                story = [b for b in SHOW[n].blocks if b.kind != "camera"]
                last = story[-1]
                start = next(b for b in story if b.start >= last.end - 25 and b.kind == "dialogue")
                out.append({"id": cid, "candidate": {"candidates": [{
                    "segments": [{"episode": n, "from": start.id, "to": last.id}],
                    "hook": {"episode": n, "block": start.id}, "cliffhanger": {"episode": n, "block": last.id}}]}})
            return json.dumps({"repairs": out})
        if "separate promo clips" in prompt:
            ids = _re.findall(r"=== CLIP (C\d+)", prompt)
            return json.dumps({"reviews": [{"id": i, "scores": {"hook_strength": 20, "standalone_clarity": 15,
                                                                 "cliffhanger_strength": 20, "emotional_intensity": 10,
                                                                 "narrative_coherence": 8}, "flags": {}} for i in ids]})
        if "passed QA" in prompt:
            return json.dumps({"per_promo": {}, "publishing_plan": "plan"})
        raise AssertionError("unexpected prompt: " + prompt[:200])


class TestBatchMode(unittest.TestCase):
    def test_full_pack_fits_free_quota(self):
        tmp = tempfile.mkdtemp()
        try:
            scout_llm = BatchFakeLLM("scout", cache_dir=None)
            critic_llm = BatchFakeLLM("critic", cache_dir=None)
            run = Run(DATA, os.path.join(tmp, "b"), scout_llm, critic_llm=critic_llm, batch_episodes=10,
                      batch_critic=15, max_arc_ideas=5, reference=None,
                      probes=os.path.join(ROOT, "eval", "critic_probes.json"), log=lambda m: None)
            doc = run.execute()
            self.assertEqual(run.failures, [])
            self.assertEqual(len(doc["analysis_summary"]["episodes_analyzed"]), 50)
            # 5 scout + 1 planner + 1 arc build + 1 repair + 1 ranking
            self.assertLessEqual(scout_llm.calls, 20, "scout model must fit a 20 requests/day free quota")
            self.assertLessEqual(critic_llm.calls, 20, "critic model must fit a 20 requests/day free quota")
            self.assertTrue(any(p["origin"].startswith("arc") for p in doc["ranked_promos"]))
            self.assertTrue(any(r["stage_rejected"] == "validation" for r in doc["rejected_candidates"]))
            rows = [json.loads(l) for l in open(os.path.join(tmp, "b", "candidates_all.jsonl"))]
            self.assertTrue(any(r["repaired_from"] for r in rows), "wrapped repair answers must be parsed")
            reviewed = [c for c in doc["ranked_promos"]]
            self.assertGreater(len(reviewed), 30)
            errors, _ = check(DATA, doc)
            self.assertEqual(errors, [])
            print(f"\n  batch mode: scout model {scout_llm.calls} requests, critic model {critic_llm.calls} requests")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Audit regressions (v0.2): visibility, description support, review flags, counts
# ---------------------------------------------------------------------------
from promo_finder.review import review, description_support  # noqa: E402


def _crit(**over):
    scores = {"hook_strength": 20, "standalone_clarity": 16, "cliffhanger_strength": 20,
              "emotional_intensity": 12, "narrative_coherence": 8}
    scores.update(over)
    return {"scores": scores, "flags": {}}


class TestReviewFlags(unittest.TestCase):
    def _p010(self, cliff_block=26, cliff_desc="Alexander grabs Juliana's hand to pull her through the burning corridors"):
        # Real case: E18 #2-#26. #26 is a CAMERA shot starting exactly at the cut (41.4 s).
        c = Candidate("P010", "test", [Segment(18, 2, 26)], BlockRef(18, 2),
                      "Juliana wakes up gasping and coughing as fire glows beneath her bedroom door",
                      BlockRef(18, cliff_block), cliff_desc, "")
        return validate(SHOW, c)

    def test_cliffhanger_shot_after_cut_is_not_on_screen(self):
        c = self._p010()
        self.assertTrue(any("not on screen" in e for e in c.errors), c.errors)
        apply_critique(c, _crit())
        r = review(SHOW, c)
        self.assertEqual(r["proposed_fixes"]["cliffhanger"]["block"], 25)
        self.assertIn("another way", r["proposed_fixes"]["cliffhanger"]["description"])

    def test_displayed_times_never_exceed_the_cut(self):
        from promo_finder.candidates import timeline
        c = self._p010(cliff_block=25, cliff_desc="Then we go another way.")
        self.assertEqual(c.errors, [])
        self.assertLessEqual(max(t[1] for t in timeline(SHOW, c)), c.total_duration)

    def test_hook_absent_from_clip(self):
        s = segment_from_times(SHOW, 3, "00:58.30", "01:25.50")
        c = validate(SHOW, Candidate("x", "t", [Segment(3, s.from_block, s.to_block)], BlockRef(3, 0), "", BlockRef(3, s.to_block), "", ""))
        self.assertTrue(any("hook block" in e and "not on screen" in e for e in c.errors), c.errors)
        apply_critique(c, _crit())
        self.assertTrue(any(f["code"] == "hook_not_on_screen" for f in review(SHOW, c)["flags"]))

    def test_hook_after_three_seconds(self):
        s = segment_from_times(SHOW, 1, "00:11.30", "00:56.90")
        late = next(b.id for b in SHOW[1].blocks if b.start >= 20 and b.kind == "dialogue")
        c = validate(SHOW, Candidate("x", "t", [Segment(1, s.from_block, s.to_block)], BlockRef(1, late), "",
                                     BlockRef(1, s.to_block), "", ""))
        self.assertTrue(any("outside the first 3s" in e for e in c.errors))
        apply_critique(c, _crit())
        r = review(SHOW, c)
        self.assertTrue(any(f["code"] == "hook_late" and f["action"] == "reject" for f in r["flags"]))
        self.assertEqual(r["status"], "not_recommended")

    def test_unsupported_cliffhanger_description(self):
        c = self._p010(cliff_block=25, cliff_desc="The ceiling beams collapse and bury the chapel in flames")
        apply_critique(c, _crit())
        flags = review(SHOW, c)["flags"]
        f = next(f for f in flags if f["code"] == "cliffhanger_description_unsupported")
        self.assertEqual((f["kind"], f["action"]), ("approximate", "inspect"))

    def test_supported_description_not_flagged(self):
        c = self._p010(cliff_block=25, cliff_desc="Trapped by the fire, Juliana cries they cannot get through; Alexander: another way")
        apply_critique(c, _crit())
        self.assertFalse(any(f["code"].endswith("description_unsupported") for f in review(SHOW, c)["flags"]))
        self.assertGreaterEqual(description_support("cannot get through another way", ['"We cannot get through!"']), 0.5)

    def test_zero_coherence_goes_to_human_review_not_silent_pass(self):
        c = self._p010(cliff_block=25, cliff_desc="Then we go another way.")
        apply_critique(c, _crit(narrative_coherence=0))
        self.assertEqual(c.rejection_reasons, [])          # not auto-rejected ...
        r = review(SHOW, c)
        self.assertTrue(any(f["code"] == "zero_coherence" for f in r["flags"]))
        self.assertEqual(r["status"], "needs_review")       # ... but not presented as a clean recommendation

    def test_malformed_score_key_is_normalised_not_zeroed(self):
        c = self._p010(cliff_block=25, cliff_desc="Then we go another way.")
        crit = _crit()
        crit["scores"]["narrative_ coherence"] = crit["scores"].pop("narrative_coherence")   # real Gemini output
        apply_critique(c, crit)
        self.assertEqual(c.scores["narrative_coherence"], 8)
        self.assertEqual(c.critique["missing_scores"], [])

    def test_missing_score_flagged_not_used_to_reject(self):
        c = self._p010(cliff_block=25, cliff_desc="Then we go another way.")
        crit = _crit()
        del crit["scores"]["hook_strength"]
        apply_critique(c, crit)
        self.assertFalse(any("hook score" in r for r in c.rejection_reasons))
        self.assertTrue(any(f["code"] == "critic_score_missing" for f in review(SHOW, c)["flags"]))

    def test_scout_critic_disagreement(self):
        c = self._p010(cliff_block=25, cliff_desc="Then we go another way.")
        c.self_scores = {"hook_strength": 25, "standalone_clarity": 20, "cliffhanger_strength": 25,
                         "emotional_intensity": 15, "narrative_coherence": 10}
        apply_critique(c, _crit())
        f = next(f for f in review(SHOW, c)["flags"] if f["code"] == "scout_critic_disagreement")
        self.assertEqual(f["kind"], "subjective")

    def test_cross_episode_story_order(self):
        a = segment_from_times(SHOW, 27, "00:17.20", "00:31.40")
        b = segment_from_times(SHOW, 26, "01:30.20", "01:55.50")
        c = validate(SHOW, Candidate("x", "t", [Segment(27, a.from_block, a.to_block), Segment(26, b.from_block, b.to_block)],
                                     BlockRef(27, a.from_block), "", BlockRef(26, b.to_block), "", ""))
        self.assertTrue(any("backwards" in w for w in c.warnings))
        apply_critique(c, _crit())
        codes = {f["code"]: f["action"] for f in review(SHOW, c)["flags"]}
        self.assertEqual(codes.get("story_order"), "reject")
        self.assertIn("cross_episode", codes)


class TestCountsAndAudit(unittest.TestCase):
    def test_live_calls_vs_cache_hits(self):
        tmp = tempfile.mkdtemp()
        try:
            cache = os.path.join(tmp, "cache")
            m = []
            for i in (1, 2):
                llm = FakeLLM("fake-model", cache_dir=cache)
                Run(DATA, os.path.join(tmp, f"r{i}"), llm, episodes=[3, 26, 27], workers=2, reference=None,
                    probes=None, log=lambda x: None).execute()
                with open(os.path.join(tmp, f"r{i}", "manifest.json")) as f:
                    m.append(json.load(f))
            self.assertGreater(m[0]["live_model_calls"], 0)
            self.assertEqual(m[0]["cache_hits"], 0)
            self.assertEqual(m[1]["live_model_calls"], 0)
            self.assertEqual(m[1]["cache_hits"], m[0]["live_model_calls"])
            self.assertEqual(m[0]["recorded_responses"], m[1]["recorded_responses"])
            self.assertEqual(m[1]["live_model_calls"] + m[1]["cache_hits"], m[1]["recorded_responses"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_audit_leaves_original_untouched_and_verifies(self):
        import hashlib
        from promo_finder.audit import run_audit
        tmp = tempfile.mkdtemp()
        try:
            run = os.path.join(tmp, "run")
            Run(DATA, run, None, episodes=list(range(1, 11)), reference=None, probes=None, log=lambda x: None).execute()
            digest = lambda: {f: hashlib.sha256(open(os.path.join(run, f), "rb").read()).hexdigest()  # noqa: E731
                              for f in ("promos.json", "manifest.json", "candidates_all.jsonl")}
            before = digest()
            doc = run_audit(DATA, run, os.path.join(tmp, "audited"))
            self.assertEqual(before, digest())
            self.assertEqual(doc["audit"]["model_calls_made_by_audit"], 0)
            self.assertTrue(all(c["ok"] for c in doc["audit"]["reconciliation"]["checks"]))
            self.assertEqual(check(DATA, doc)[0], [])
            self.assertTrue(all("review" in p and "original_rank" in p for p in doc["ranked_promos"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)



# ---------------------------------------------------------------------------
# v0.3 regressions: deterministic repair, borderline policy, review reuse, gap pass, compare
# ---------------------------------------------------------------------------
from promo_finder.repair import trim_to_rules  # noqa: E402


class TestDeterministicRepair(unittest.TestCase):
    def test_trims_clip_that_runs_past_its_cliffhanger(self):
        # Same failure as the missed ring scene (ep 26): the clip runs on more than 4 s after the
        # declared cliffhanger ("Do not move!"), so validation fails; trimming the end fixes it.
        a = segment_from_times(SHOW, 26, "01:30.20", "01:55.50")
        cliff = next(b.id for b in SHOW[26].blocks if b.text == '"Do not move!"')
        c = validate(SHOW, Candidate("ring", "t", [Segment(26, a.from_block, a.to_block)], BlockRef(26, a.from_block),
                                     "", BlockRef(26, cliff), "", ""))
        self.assertTrue(any("past the cliffhanger" in e for e in c.errors), c.errors)
        t = trim_to_rules(SHOW, c)
        self.assertIsNotNone(t)
        self.assertEqual(t.errors, [])
        self.assertLess(t.segments[0].to_block, a.to_block)
        self.assertEqual(t.cliffhanger.block, cliff)     # the model's cliffhanger choice is kept
        self.assertTrue(any("no model call" in w for w in t.warnings))

    def test_trims_start_so_hook_is_in_first_three_seconds(self):
        s = segment_from_times(SHOW, 1, "00:11.30", "00:56.90")
        hook = next(b.id for b in SHOW[1].blocks if "Why are you holding her hand" in b.text)
        c = validate(SHOW, Candidate("x", "t", [Segment(1, s.from_block, s.to_block)], BlockRef(1, hook), "",
                                     BlockRef(1, s.to_block), "", ""))
        self.assertTrue(any("outside the first 3s" in e for e in c.errors))
        t = trim_to_rules(SHOW, c)
        self.assertIsNotNone(t)
        self.assertLessEqual(t.hook_offset, 3.0)
        self.assertEqual(t.hook.block, hook)            # the model's hook choice is kept

    def test_retargets_offscreen_cliffhanger_to_last_beat(self):
        c = validate(SHOW, Candidate("P010", "t", [Segment(18, 2, 26)], BlockRef(18, 2), "", BlockRef(18, 26),
                                     "beams collapse", ""))
        t = trim_to_rules(SHOW, c)
        self.assertIsNotNone(t)
        self.assertEqual(t.cliffhanger.block, 25)
        self.assertIn("another way", t.cliffhanger_description)

    def test_gives_up_when_rules_cannot_be_met_by_trimming(self):
        s = segment_from_times(SHOW, 1, "00:11.30", "01:30.80")     # 79.5 s, hook at start, cliffhanger at end
        c = validate(SHOW, Candidate("x", "t", [Segment(1, s.from_block, s.to_block)], BlockRef(1, s.from_block), "",
                                     BlockRef(1, s.to_block), "", ""))
        self.assertTrue(any("exceeds 60" in e for e in c.errors))
        self.assertIsNone(trim_to_rules(SHOW, c))


class TestBorderlinePolicy(unittest.TestCase):
    def _c(self):
        return candidate_from_times(SHOW, "ok", [(3, "00:58.30", "01:25.50"), (3, "01:59.80", "02:25.75")])

    def test_one_weak_dimension_on_good_clip_goes_to_review(self):
        c = apply_critique(self._c(), _crit(hook_strength=12, standalone_clarity=18, cliffhanger_strength=23,
                                             emotional_intensity=15, narrative_coherence=10))
        self.assertEqual(c.rejection_reasons, [])
        self.assertTrue(c.borderline_reasons)
        self.assertEqual(review(SHOW, c)["status"], "needs_review")

    def test_strict_policy_still_rejects(self):
        c = apply_critique(self._c(), _crit(hook_strength=12, standalone_clarity=18, cliffhanger_strength=23,
                                             emotional_intensity=15, narrative_coherence=10), policy="strict")
        self.assertTrue(c.rejection_reasons)

    def test_two_weak_dimensions_reject(self):
        c = apply_critique(self._c(), _crit(hook_strength=12, cliffhanger_strength=12, standalone_clarity=20,
                                             emotional_intensity=15, narrative_coherence=10))
        self.assertTrue(c.rejection_reasons)

    def test_very_weak_hook_rejects(self):
        c = apply_critique(self._c(), _crit(hook_strength=5, standalone_clarity=20, cliffhanger_strength=25,
                                             emotional_intensity=15, narrative_coherence=10))
        self.assertTrue(c.rejection_reasons)

    def test_weak_dimension_on_mediocre_clip_rejects(self):
        c = apply_critique(self._c(), _crit(hook_strength=12, standalone_clarity=14, cliffhanger_strength=16,
                                             emotional_intensity=10, narrative_coherence=8))
        self.assertTrue(c.rejection_reasons)
        self.assertFalse(c.borderline_reasons)

    def test_misleading_edit_goes_to_review_not_auto_pass(self):
        c = apply_critique(self._c(), {"scores": {"hook_strength": 20, "standalone_clarity": 15,
                                                  "cliffhanger_strength": 23, "emotional_intensity": 15,
                                                  "narrative_coherence": 4}, "flags": {"misleading_edit": True}})
        self.assertEqual(c.rejection_reasons, [])
        self.assertTrue(any("misleading" in r for r in c.borderline_reasons))


class TestReviewReuse(unittest.TestCase):
    def test_reuses_only_identical_block_sequences(self):
        from promo_finder.memo import CritiqueMemo
        tmp = tempfile.mkdtemp()
        try:
            c = candidate_from_times(SHOW, "a", [(3, "00:58.30", "01:25.50")])
            clip = render_clip(SHOW, c)
            prompt = f"=== CLIP C1 ({c.total_duration:.1f} s) ===\n{clip}\n\nFor EACH clip evaluate"
            with open(os.path.join(tmp, "x.json"), "w") as f:
                json.dump({"provider": "p", "model": "m", "effort": None, "prompt": prompt,
                           "response": json.dumps({"reviews": [{"id": "C1", "scores": {"hook_strength": 21}}]})}, f)
            memo = CritiqueMemo(tmp, "p", "m", None, SHOW)
            self.assertEqual(memo.get(c)["scores"]["hook_strength"], 21)
            other = candidate_from_times(SHOW, "b", [(3, "00:58.30", "01:28.00")])   # one more block shown
            self.assertIsNone(memo.get(other))
            self.assertEqual(len(CritiqueMemo(tmp, "p", "other-model", None, SHOW).reviews), 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestGapPassAndCompare(unittest.TestCase):
    def test_gap_pass_and_compare(self):
        from promo_finder.compare import compare
        tmp = tempfile.mkdtemp()
        try:
            out = os.path.join(tmp, "g")
            scout_llm = BatchFakeLLM("scout", cache_dir=None)
            critic_llm = BatchFakeLLM("critic", cache_dir=None)
            run = Run(DATA, out, scout_llm, critic_llm=critic_llm, batch_episodes=10, batch_critic=15,
                      gap_pass=True, reference=None, probes=None, log=lambda m: None)
            doc = run.execute()
            self.assertEqual(run.failures, [])
            self.assertEqual(check(DATA, doc)[0], [])
            m = json.load(open(os.path.join(out, "manifest.json")))
            self.assertIn("gap_pass", m)
            res = compare(DATA, [out], os.path.join(ROOT, "eval", "reference_picks.json"))[0]
            self.assertEqual(res["references"], 17)
            self.assertLessEqual(res["recall_recommended"], res["recall_all"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_verifier_requires_clean_promos_before_borderline(self):
        doc = Run(DATA, tempfile.mkdtemp(), None, episodes=[1, 2, 3], reference=None, probes=None,
                  log=lambda x: None).execute()
        if len(doc["ranked_promos"]) >= 2:
            doc["ranked_promos"][0]["borderline"] = True
            errors, _ = check(DATA, doc)
            self.assertTrue(any("clean promo is ranked below a borderline" in e for e in errors))


class TestUsageCounting(unittest.TestCase):
    def test_fallback_model_calls_are_counted(self):
        # v0.3 manifest bug: a request answered by the fallback model was missing from live_model_calls.
        main, critic, fallback = (FakeLLM(m, cache_dir=None) for m in ("main", "critic", "fallback"))
        main.calls, critic.calls, fallback.calls = 5, 3, 1
        run = Run(DATA, tempfile.mkdtemp(), main, critic_llm=critic, fallback_llm=fallback, reference=None,
                  probes=None, log=lambda m: None)
        self.assertEqual(run._usage("calls"), 9)
        same = Run(DATA, tempfile.mkdtemp(), main, critic_llm=main, fallback_llm=main, reference=None, probes=None,
                   log=lambda m: None)
        self.assertEqual(same._usage("calls"), 5)          # one model object is counted once

    def test_repeated_stage_name_keeps_every_raw_response(self):
        # v0.3.1 bug: the gap pass reused "repair_batch01" / "critic_batch01_01", overwriting raw files.
        out = tempfile.mkdtemp()
        try:
            run = Run(DATA, out, FakeLLM("m", cache_dir=None), reference=None, probes=None, log=lambda m: None)
            run.llm.complete = lambda system, prompt: "reply to " + prompt
            run.ask("repair_batch01", "sys", "first")
            run.ask("repair_batch01", "sys", "second")
            raw = os.path.join(out, "raw")
            self.assertEqual(json.load(open(os.path.join(raw, "repair_batch01.json")))["prompt"], "first")
            self.assertEqual(json.load(open(os.path.join(raw, "repair_batch01_2.json")))["prompt"], "second")
        finally:
            shutil.rmtree(out, ignore_errors=True)


class TestDedupPrefersCleanClips(unittest.TestCase):
    def test_flagged_clip_does_not_displace_clean_overlap(self):
        # v4 lost reference R03: a flagged clip (82) displaced an overlapping recommended one (80).
        from promo_finder.pipeline import dedup
        from promo_finder.candidates import Candidate, Segment, BlockRef
        mk = lambda cid, a, b: Candidate(cid, "t", [Segment(12, a, b)], BlockRef(12, a), "",
                                                        BlockRef(12, b), "", "")
        flagged, clean, other = mk("flagged", 20, 50), mk("clean", 23, 54), mk("other", 0, 10)
        flagged.total_score, clean.total_score, other.total_score = 82, 80, 70
        passing = [flagged, clean, other]          # ranking order: score first
        reviews = {id(flagged): {"status": "needs_review"}, id(clean): {"status": "recommended"},
                   id(other): {"status": "recommended"}}
        with unittest.mock.patch("promo_finder.pipeline.overlap_seconds",
                                 lambda a, b: 30.0 if {a.id, b.id} == {"flagged", "clean"} else 0.0), \
             unittest.mock.patch.object(Candidate, "total_duration", property(lambda c: 30.0)):
            final = dedup(passing, reviews)
        self.assertEqual([c.id for c in final], ["clean", "other"])
        self.assertEqual(flagged.duplicate_of, "clean")

    def test_ranking_order_is_kept(self):
        from promo_finder.pipeline import dedup
        from promo_finder.candidates import Candidate, Segment, BlockRef
        cs = [Candidate(f"c{i}", "t", [Segment(1, i, i)], BlockRef(1, i), "", BlockRef(1, i), "", "") for i in range(3)]
        reviews = {id(cs[0]): {"status": "needs_review"}, id(cs[1]): {"status": "recommended"},
                   id(cs[2]): {"status": "recommended"}}
        with unittest.mock.patch("promo_finder.pipeline.overlap_seconds", lambda a, b: 0.0), \
             unittest.mock.patch.object(Candidate, "total_duration", property(lambda c: 30.0)):
            self.assertEqual([c.id for c in dedup(cs, reviews)], ["c0", "c1", "c2"])


class TestNewShow(unittest.TestCase):
    def test_eval_files_for_another_show_give_a_clear_error(self):
        from promo_finder.fixtures import load_fixture_file
        small = {n: SHOW[n] for n in (1, 2)}               # a "show" without the reference episodes
        with self.assertRaisesRegex(ValueError, "is for another show.*--reference ''"):
            load_fixture_file(small, os.path.join(ROOT, "eval", "reference_picks.json"), origin="reference")
        self.assertEqual(len(load_fixture_file(SHOW, os.path.join(ROOT, "eval", "reference_picks.json"),
                                               origin="reference")), 17)
