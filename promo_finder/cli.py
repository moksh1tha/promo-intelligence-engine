"""Command line entry point.

  python -m promo_finder run --data data --out runs/opus                 # Claude (default)
  python -m promo_finder run --provider gemini                           # free tier (GEMINI_API_KEY)
  python -m promo_finder run --provider groq --model <id>                # free tier (GROQ_API_KEY)
  python -m promo_finder run --provider openai --model <name> --base-url <url>
  python -m promo_finder models --provider gemini                        # list model ids
  python -m promo_finder run --provider heuristic                        # offline baseline
  python -m promo_finder run --provider replay:anthropic --out runs/x    # re-run from cache only
  python -m promo_finder verify --data data --promos runs/opus/promos.json
  python -m promo_finder audit --run runs/gemini-batch                   # no model calls
"""
import argparse
import datetime as dt
import os
import sys

from .llm import LLMError, PRESETS, make_llm


def load_dotenv(path: str = ".env"):
    """Load KEY=VALUE lines from .env into the environment (without overriding real env vars)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k.startswith("export "):
                k = k[len("export "):].strip()
            os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def _episodes(spec):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="promo_finder", description="Find promo clips in microdrama screenplays.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the full pipeline")
    r.add_argument("--data", default="data", help="folder of episode .txt screenplays")
    r.add_argument("--out", default=None, help="output folder (default runs/<provider>-<timestamp>)")
    r.add_argument("--provider", default=os.environ.get("PROMO_PROVIDER", "anthropic"),
                   help="anthropic | gemini | groq | openrouter | ollama | openai | heuristic | replay:<provider>")
    r.add_argument("--model", default=os.environ.get("PROMO_MODEL"), help="model id (default per provider)")
    r.add_argument("--effort", default=os.environ.get("PROMO_EFFORT"),
                   help="thinking depth: Claude low|medium|high|xhigh|max (default high); "
                        "Gemini/OpenAI reasoning models low|medium|high (default low for gemini)")
    r.add_argument("--base-url", default=None, help="override the endpoint URL (OpenAI-compatible providers)")
    r.add_argument("--rpm", type=float, default=None,
                   help="max requests per minute (default per provider; free tiers are paced conservatively)")
    r.add_argument("--cache", default=".cache/llm", help="response cache folder ('' to disable)")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--episodes", default=None, help="subset, e.g. 1-10,12")
    r.add_argument("--per-episode", type=int, default=3, help="max candidates the scout returns per episode")
    r.add_argument("--arc-ideas", type=int, default=8, help="max cross-episode ideas (0 disables)")
    r.add_argument("--batch", action="store_true",
                   help="low-quota mode: 10 episodes per scout request, 15 clips per critic request, "
                        "5 cross-episode ideas built in one request (~20-25 requests for 50 episodes)")
    r.add_argument("--batch-episodes", type=int, default=None, help="episodes per scout request (implies batching)")
    r.add_argument("--batch-critic", type=int, default=None, help="clips per critic request (implies batching)")
    r.add_argument("--critic-model", default=os.environ.get("PROMO_CRITIC_MODEL"),
                   help="different model for the critic, same provider (e.g. to use a second free quota)")
    r.add_argument("--critic-policy", choices=["borderline", "strict"], default="borderline",
                   help="borderline (default): one weak subjective score on an otherwise good clip -> human review; "
                        "strict: any weak score rejects (v0.2 behaviour)")
    r.add_argument("--gap-pass", action="store_true",
                   help="batch mode: give episodes with no surviving clip one more, broader look (~1 request per 10 such episodes)")
    r.add_argument("--arc-planner", type=int, choices=[1, 2], default=2,
                   help="2 (default): planner also sees each episode's first/last line and is asked for setup-payoff, "
                        "recognition and continuation ideas; 1: v0.2 planner")
    r.add_argument("--fallback-model", default=os.environ.get("PROMO_FALLBACK_MODEL"),
                   help="same provider: answers main-model requests when its daily quota is exhausted")
    r.add_argument("--no-reuse-critiques", action="store_true",
                   help="do not reuse earlier critic reviews of identical clips from the cache")
    r.add_argument("--reference", default="eval/reference_picks.json", help="reference picks for recall ('' to skip)")
    r.add_argument("--probes", default="eval/critic_probes.json", help="critic calibration probes ('' to skip)")

    m = sub.add_parser("models", help="list the model ids an OpenAI-compatible provider serves")
    m.add_argument("--provider", default="gemini", choices=sorted(PRESETS))
    m.add_argument("--base-url", default=None)

    a = sub.add_parser("audit", help="re-check a finished run (no model calls) and write an audited copy")
    a.add_argument("--data", default="data")
    a.add_argument("--run", required=True, help="run folder to audit (left unchanged)")
    a.add_argument("--out", default=None, help="output folder (default <run>-audited)")

    cp = sub.add_parser("compare", help="compare runs with one evaluation method (no model calls)")
    cp.add_argument("runs", nargs="+")
    cp.add_argument("--data", default="data")
    cp.add_argument("--reference", default="eval/reference_picks.json")
    cp.add_argument("--out", default=None, help="also write the table to this markdown file")

    v = sub.add_parser("verify", help="re-check a promos.json against the screenplays")
    v.add_argument("--data", default="data")
    v.add_argument("--promos", required=True)

    args = ap.parse_args(argv)
    load_dotenv()

    if args.cmd == "models":
        try:
            llm = make_llm(args.provider, "list-models", None, base_url=args.base_url)
            for mid in llm.list_models():
                print(mid)
        except LLMError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.cmd == "audit":
        from .audit import run_audit
        from .verify import verify_file
        out = args.out or args.run.rstrip("/") + "-audited"
        doc = run_audit(args.data, args.run, out)
        statuses = {}
        for p in doc["ranked_promos"]:
            statuses[p["review"]["status"]] = statuses.get(p["review"]["status"], 0) + 1
        print(f"audit: {len(doc['ranked_promos'])} promos -> {out}  {statuses}", file=sys.stderr)
        return 0 if verify_file(args.data, os.path.join(out, "promos.json")) else 1

    if args.cmd == "compare":
        from .compare import compare, to_markdown
        md = to_markdown(compare(args.data, args.runs, args.reference))
        print(md)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(md + "\n")
        return 0

    if args.cmd == "verify":
        from .verify import verify_file
        ok = verify_file(args.data, args.promos)
        return 0 if ok else 1

    from .pipeline import Run
    provider = args.provider
    out = args.out or os.path.join("runs", f"{provider.replace(':', '-')}-{args.model or 'default'}-"
                                           f"{dt.datetime.now():%Y%m%d-%H%M%S}".replace("/", "_"))
    critic_llm = fallback_llm = None
    if provider == "heuristic":
        llm = None
    else:
        effort = args.effort or {"anthropic": "high", "gemini": "low", "replay:anthropic": "high",
                                  "replay:gemini": "low"}.get(provider)
        try:
            llm = make_llm(provider, args.model, args.cache or None, effort=effort, base_url=args.base_url,
                           rpm=args.rpm)
            critic_llm = make_llm(provider, args.critic_model, args.cache or None, effort=effort,
                                  base_url=args.base_url, rpm=args.rpm) if args.critic_model else None
            fallback_llm = make_llm(provider, args.fallback_model, args.cache or None, effort=effort,
                                    base_url=args.base_url, rpm=args.rpm) if args.fallback_model else None
        except LLMError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    batch_eps = args.batch_episodes if args.batch_episodes is not None else (10 if args.batch else 0)
    batch_crit = args.batch_critic if args.batch_critic is not None else (15 if args.batch else 0)
    arc_ideas = args.arc_ideas   # batch mode builds all ideas in one request, so more ideas cost no extra calls
    run = Run(args.data, out, llm, workers=args.workers, max_per_episode=args.per_episode,
              max_arc_ideas=arc_ideas, episodes=_episodes(args.episodes),
              critic_llm=critic_llm, batch_episodes=batch_eps, batch_critic=batch_crit,
              critic_policy=args.critic_policy, gap_pass=args.gap_pass, arc_planner_version=args.arc_planner,
              fallback_llm=fallback_llm, reuse_critiques=not args.no_reuse_critiques,
              reference=args.reference or None, probes=args.probes or None,
              log=lambda m: print(m, file=sys.stderr, flush=True))
    try:
        run.execute()
    except FileNotFoundError as exc:
        print(f"error: {exc} (copy the input pack's .txt files there, or pass --data <folder>)", file=sys.stderr)
        return 2
    if run.failures and any(x is not None and getattr(x, "quota_exhausted", None) for x in (llm, critic_llm)):
        print("\nThe provider's quota ran out during this run, so some calls failed (see manifest.json).\n"
              "Re-run the same command after the quota resets: finished calls come from the cache.",
              file=sys.stderr)
    from .verify import verify_file
    ok = verify_file(args.data, os.path.join(out, "promos.json"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
