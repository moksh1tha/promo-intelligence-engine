"""Swappable text-model backends.

Every model call in the pipeline goes through `LLM.complete(system, prompt)`
and returns plain text. Adding a provider means adding one small class.

  anthropic  - Claude via the official `anthropic` SDK (default model claude-opus-5-5)
  openai     - any OpenAI-compatible chat endpoint, plain HTTP (no package needed):
  gemini     -   preset: Google AI Studio (free tier available)
  groq       -   preset: Groq (free tier available)
  openrouter -   preset: OpenRouter (free ":free" models available)
  ollama     -   preset: local Ollama server (free, runs on your machine)
  replay     - answers only from the response cache; fails on a cache miss.
               Used to re-run a pipeline exactly from saved raw outputs.

Free tiers have per-minute and per-day limits, so every backend can be paced
(`rpm`), honours server retry hints on HTTP 429, and stops early when a daily
quota is exhausted. Because responses are cached, re-running after the quota
resets continues where the last run stopped.

Responses are cached on disk keyed by (provider, model, effort, system, prompt),
so re-runs are free and reproducible.
"""
import hashlib
import json
import os
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import List, Optional


class LLMError(RuntimeError):
    pass


class RetryableError(LLMError):
    def __init__(self, msg: str, retry_after: Optional[float] = None):
        super().__init__(msg)
        self.retry_after = retry_after


class ReplayMiss(LLMError):
    """Replay mode: the cache has no response for this prompt."""


class QuotaExhausted(LLMError):
    """Daily / hard quota reached. Retrying now will not help; re-run later (cache resumes the run)."""


class LLM:
    provider = "base"

    def __init__(self, model: str, cache_dir: Optional[str] = None, effort: Optional[str] = None,
                 max_tokens: int = 16000, rpm: float = 0):
        self.model = model
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._next_slot = 0.0
        self._slot_lock = threading.Lock()
        self.quota_exhausted: Optional[str] = None
        self.effort = effort
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self.calls = 0
        self.cache_hits = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ---- public -------------------------------------------------------
    def complete(self, system: str, prompt: str) -> str:
        key = self._key(system, prompt)
        cached = self._cache_get(key)
        if cached is not None:
            with self._lock:
                self.cache_hits += 1
            return cached
        if self.quota_exhausted:
            raise QuotaExhausted(self.quota_exhausted)
        text = self._with_retries(lambda: self._paced_call(system, prompt))
        with self._lock:
            self.calls += 1
        self._cache_put(key, system, prompt, text)
        return text

    def describe(self) -> str:
        return f"{self.provider}:{self.model}" + (f" (effort={self.effort})" if self.effort else "")

    # ---- to implement -------------------------------------------------
    def _call(self, system: str, prompt: str) -> str:
        raise NotImplementedError

    def _is_retryable(self, exc: Exception) -> bool:
        return False

    # ---- helpers ------------------------------------------------------
    def _paced_call(self, system, prompt):
        if self.min_interval:
            with self._slot_lock:
                now = time.monotonic()
                wait = self._next_slot - now
                self._next_slot = max(now, self._next_slot) + self.min_interval
            if wait > 0:
                time.sleep(wait)
        return self._call(system, prompt)

    def _with_retries(self, fn, attempts: int = 8):
        for i in range(attempts):
            try:
                return fn()
            except QuotaExhausted as exc:
                self.quota_exhausted = str(exc)
                raise
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise their own types
                retryable = isinstance(exc, RetryableError) or self._is_retryable(exc)
                if i == attempts - 1 or not retryable:
                    raise
                hint = getattr(exc, "retry_after", None)
                time.sleep(min(120.0, hint + random.random()) if hint else min(60.0, 2 ** i + random.random()))

    def _key(self, system: str, prompt: str) -> str:
        h = hashlib.sha256()
        for part in (self.provider, self.model, self.effort or "", system, prompt):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, key + ".json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)["response"]   # a cache entry is {"provider", "model", ..., "prompt", "response"}
        return None

    def _cache_put(self, key, system, prompt, text):
        if not self.cache_dir:
            return
        path = os.path.join(self.cache_dir, key + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"provider": self.provider, "model": self.model, "effort": self.effort,
                       "system_sha": hashlib.sha256(system.encode()).hexdigest(),
                       "prompt": prompt, "response": text}, f, ensure_ascii=False, indent=1)


class AnthropicLLM(LLM):
    provider = "anthropic"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise LLMError("pip install anthropic  (needed for --provider anthropic)") from exc
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY / ant auth profile

    def _call(self, system: str, prompt: str) -> str:
        # output_config / fallbacks go through extra_body so the adapter also works with
        # older 0.x SDK releases (the only ones installable on Python 3.9).
        extra = {}
        if self.effort:
            extra["output_config"] = {"effort": self.effort}
        betas = []
        if os.environ.get("PROMO_NO_FALLBACK") != "1":
            # Server-side refusal fallback: if a safety classifier declines (these scripts contain
            # violence and abuse), the API re-runs the request on Anthropic's recommended fallback model.
            extra["fallbacks"] = "default"
            betas.append("server-side-fallback-2026-07-01")
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            betas=betas,
            extra_body=extra,
        )
        if resp.stop_reason == "refusal":
            raise LLMError(f"model refused: {getattr(resp, 'stop_details', None)}")
        if resp.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")
        usage = getattr(resp, "usage", None)
        if usage is not None:
            with self._lock:
                self.input_tokens += getattr(usage, "input_tokens", 0) or 0
                self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def _is_retryable(self, exc):
        if isinstance(exc, self._anthropic.APIConnectionError):
            return True
        status = getattr(exc, "status_code", None)
        return status is not None and (status in (408, 409, 429) or status >= 500)


# name: (base_url, api-key env vars, default model, default requests/minute)
PRESETS = {
    "openai": ("https://api.openai.com/v1", ["OPENAI_API_KEY"], None, 0),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
               "gemini-flash-latest", 10),
    "groq": ("https://api.groq.com/openai/v1", ["GROQ_API_KEY"], None, 20),
    "openrouter": ("https://openrouter.ai/api/v1", ["OPENROUTER_API_KEY"], None, 15),
    "ollama": ("http://localhost:11434/v1", [], None, 0),
}

_DAILY = re.compile(r"per ?day|perday|daily|RequestsPerDay|quota exceeded for metric.*day|insufficient_quota|"
                    r"credits", re.I)


class OpenAICompatibleLLM(LLM):
    """Any OpenAI-style /chat/completions endpoint, over plain HTTPS (stdlib only)."""

    def __init__(self, model: str, base_url: str, api_key: Optional[str], provider: str = "openai", **kwargs):
        super().__init__(model, **kwargs)
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = float(os.environ.get("PROMO_HTTP_TIMEOUT", "300"))

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")[:2000]
            status = exc.code
            if status == 429 or status >= 500 or status == 408:
                if status == 429 and _DAILY.search(text):
                    raise QuotaExhausted(f"{self.provider}: quota exhausted (HTTP 429): {text[:300]}")
                raise RetryableError(f"{self.provider}: HTTP {status}: {text[:300]}", _retry_hint(exc, text))
            raise LLMError(f"{self.provider}: HTTP {status}: {text[:500]}")
        except urllib.error.URLError as exc:
            raise RetryableError(f"{self.provider}: connection error: {exc.reason}")
        except (TimeoutError, ConnectionError, socket.timeout) as exc:   # socket.timeout is separate on Python 3.9
            raise RetryableError(f"{self.provider}: {type(exc).__name__}: {exc}")

    def _call(self, system: str, prompt: str) -> str:
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if self.effort:
            body["reasoning_effort"] = self.effort   # thinking depth on Gemini / OpenAI reasoning models
        data = self._request("POST", "/chat/completions", body)
        usage = data.get("usage") or {}
        with self._lock:
            self.input_tokens += usage.get("prompt_tokens") or 0
            self.output_tokens += usage.get("completion_tokens") or 0
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError):
            raise LLMError(f"{self.provider}: unexpected response: {str(data)[:300]}")
        if choice.get("finish_reason") in ("content_filter", "safety"):
            raise LLMError(f"{self.provider}: response blocked by the provider's safety filter")
        return (choice.get("message") or {}).get("content") or ""

    def list_models(self) -> List[str]:
        data = self._request("GET", "/models")
        return sorted(m.get("id", "") for m in data.get("data", []))


def _retry_hint(exc, body: str) -> Optional[float]:
    ra = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    m = re.search(r'retry[_ ]?delay"?\s*[:=]\s*"?(\d+(?:\.\d+)?)s', body, re.I) or \
        re.search(r"try again in (\d+(?:\.\d+)?)s", body, re.I)
    return float(m.group(1)) if m else None


class ReplayLLM(LLM):
    """Serves responses from the cache only. `model`/`provider` must match the original run."""

    def __init__(self, model: str, cache_dir: str, provider: str, effort: Optional[str] = None, **kwargs):
        super().__init__(model, cache_dir=cache_dir, effort=effort, **kwargs)
        self.provider = provider

    def _call(self, system, prompt):
        raise ReplayMiss("replay: no cached response for this prompt (prompt or model changed?)")


DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", **{k: v[2] for k, v in PRESETS.items() if v[2]}}
PROVIDERS = ["anthropic", *PRESETS, "replay:<provider>"]


def make_llm(provider: str, model: Optional[str], cache_dir: Optional[str], effort: Optional[str] = None,
             base_url: Optional[str] = None, rpm: Optional[float] = None) -> LLM:
    if provider == "anthropic":
        return AnthropicLLM(model or DEFAULT_MODELS["anthropic"], cache_dir=cache_dir, effort=effort, rpm=rpm or 0)
    if provider in PRESETS:
        url, key_envs, default_model, default_rpm = PRESETS[provider]
        if provider == "openai":
            url = base_url or os.environ.get("OPENAI_BASE_URL") or url
        elif base_url:
            url = base_url
        key = next((os.environ[k] for k in key_envs if os.environ.get(k)), None)
        if key and key.startswith("your-") and key.endswith("-here"):
            key = None   # still the placeholder from .env.example
        if key_envs and not key:
            raise LLMError(f"--provider {provider} needs an API key: set {key_envs[0]} in your shell or in .env")
        model = model or default_model
        if not model:
            raise LLMError(f"--model is required for --provider {provider} "
                           f"(see: python3 -m promo_finder models --provider {provider})")
        return OpenAICompatibleLLM(model, url, key, provider=provider, cache_dir=cache_dir, effort=effort,
                                   rpm=default_rpm if rpm is None else rpm)
    if provider.startswith("replay:"):
        orig = provider.split(":", 1)[1]
        return ReplayLLM(model or DEFAULT_MODELS.get(orig, ""), cache_dir=cache_dir, provider=orig, effort=effort)
    raise LLMError(f"unknown provider '{provider}' (choose from: {', '.join(PROVIDERS)}, heuristic)")


def extract_json(text: str):
    """Parse the first JSON object/array in a model response (tolerates code fences / prose)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        raise LLMError("no JSON found in model response")
    start = min(starts)
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(t[start:])
    return obj
