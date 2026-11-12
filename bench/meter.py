"""Usage meter: records every OpenAI call (tokens, latency, model) made inside a scope.

Why patching instead of trusting each library's own counters?
PageIndex runs an agent loop internally (several LLM turns per question, plus LLM calls
while building the tree) through LiteLLM. To compare it fairly with vector RAG we need the
*same* instrument on both sides, so we intercept at the HTTP layer (httpx), which both the
OpenAI SDK and LiteLLM use. Streaming calls are counted but carry no usage block, so both
pipelines are run non-streaming in the benchmark.

Usage:
    meter = UsageMeter(pricing)
    meter.install()
    with meter.scope("pageindex:query:q017") as s:
        ...  # any code that calls OpenAI
    print(s.summary())
"""
from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from dataclasses import dataclass, field
from typing import Any

_current: contextvars.ContextVar["Scope | None"] = contextvars.ContextVar("meter_scope", default=None)


def _get(obj: Any, *path: str, default: int = 0) -> int:
    """Read a nested attribute or dict key, tolerant of missing fields."""
    cur = obj
    for p in path:
        if cur is None:
            return default
        cur = cur.get(p) if isinstance(cur, dict) else getattr(cur, p, None)
    return int(cur) if isinstance(cur, (int, float)) else default


def parse_usage(usage: Any) -> dict:
    """Normalise usage from Chat Completions, Responses or Embeddings into one shape."""
    if usage is None:
        return {"input": 0, "cached": 0, "output": 0, "reasoning": 0}
    # Chat Completions: prompt_tokens / completion_tokens
    inp = _get(usage, "prompt_tokens") or _get(usage, "input_tokens")
    out = _get(usage, "completion_tokens") or _get(usage, "output_tokens")
    cached = _get(usage, "prompt_tokens_details", "cached_tokens") or _get(
        usage, "input_tokens_details", "cached_tokens")
    reasoning = _get(usage, "completion_tokens_details", "reasoning_tokens") or _get(
        usage, "output_tokens_details", "reasoning_tokens")
    return {"input": inp, "cached": cached, "output": out, "reasoning": reasoning}


@dataclass
class Call:
    kind: str           # chat | responses | embeddings
    model: str
    input: int
    cached: int
    output: int
    reasoning: int
    latency_s: float
    ok: bool = True


@dataclass
class Scope:
    label: str
    calls: list[Call] = field(default_factory=list)
    t_start: float = 0.0
    t_end: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, call: Call) -> None:
        with self._lock:
            self.calls.append(call)

    @property
    def wall_s(self) -> float:
        return (self.t_end or time.perf_counter()) - self.t_start

    def summary(self, pricing: dict) -> dict:
        tot = {"llm_calls": 0, "embed_calls": 0, "input_tokens": 0, "cached_tokens": 0,
               "output_tokens": 0, "reasoning_tokens": 0, "embed_tokens": 0,
               "llm_time_s": 0.0, "cost_usd": 0.0}
        for c in self.calls:
            if c.kind == "embeddings":
                tot["embed_calls"] += 1
                tot["embed_tokens"] += c.input
            else:
                tot["llm_calls"] += 1
                tot["input_tokens"] += c.input
                tot["cached_tokens"] += c.cached
                tot["output_tokens"] += c.output
                tot["reasoning_tokens"] += c.reasoning
            tot["llm_time_s"] += c.latency_s
            tot["cost_usd"] += cost_of(c, pricing)
        tot["wall_s"] = round(self.wall_s, 4)
        tot["cost_usd"] = round(tot["cost_usd"], 8)
        tot["llm_time_s"] = round(tot["llm_time_s"], 4)
        return tot


def _price_for(model: str, pricing: dict) -> dict:
    if model in pricing:
        return pricing[model]
    # Dated snapshots e.g. "gpt-5.6-terra-2026-07-09" -> longest matching prefix
    best = max((k for k in pricing if model.startswith(k)), key=len, default=None)
    if best:
        return pricing[best]
    return {"input": 0.0, "cached": 0.0, "output": 0.0, "_unknown": True}


def cost_of(c: Call, pricing: dict) -> float:
    p = _price_for(c.model, pricing)
    uncached = max(c.input - c.cached, 0)
    # reasoning tokens are already included in output/completion tokens by OpenAI
    return (uncached * p["input"] + c.cached * p.get("cached", p["input"]) + c.output * p["output"]) / 1e6


class UsageMeter:
    def __init__(self, pricing: dict):
        self.pricing = pricing
        self.unscoped = Scope("unscoped")
        self.unknown_models: set[str] = set()
        self._installed = False
        self.tracer = None      # optional bench.tracing.Tracer: one Langfuse generation per call

    # ---------------------------------------------------------------- scopes
    @contextlib.contextmanager
    def scope(self, label: str):
        s = Scope(label, t_start=time.perf_counter())
        token = _current.set(s)
        try:
            yield s
        finally:
            s.t_end = time.perf_counter()
            _current.reset(token)

    def _record(self, kind: str, kwargs: dict, resp: Any, latency: float, ok: bool) -> None:
        model = str(getattr(resp, "model", None) or kwargs.get("model") or "unknown")
        u = parse_usage(getattr(resp, "usage", None) if resp is not None else None)
        if "_unknown" in _price_for(model, self.pricing):
            self.unknown_models.add(model)
        call = Call(kind, model, u["input"], u["cached"], u["output"], u["reasoning"], latency, ok)
        (_current.get() or self.unscoped).add(call)

    # ---------------------------------------------------------------- patching
    # We patch httpx (the transport under the OpenAI SDK *and* LiteLLM, which PageIndex
    # uses). Every request to an OpenAI-style endpoint is timed and its JSON usage block
    # parsed, in the caller's own thread/task, so contextvar scopes stay correct.
    PATHS = {"/chat/completions": "chat", "/responses": "responses", "/embeddings": "embeddings"}

    def _kind(self, request) -> str | None:
        path = request.url.path
        for suffix, kind in self.PATHS.items():
            if path.endswith(suffix):
                return kind
        return None

    @staticmethod
    def _req_json(request) -> dict:
        try:
            import json
            return json.loads(request.content or b"{}")
        except Exception:
            return {}

    def _record_http(self, kind: str, req: dict, body: dict | None, latency: float, ok: bool) -> None:
        body = body or {}
        model = str(body.get("model") or req.get("model") or "unknown")
        u = parse_usage(body.get("usage"))
        if "_unknown" in _price_for(model, self.pricing):
            self.unknown_models.add(model)
        call = Call(kind, model, u["input"], u["cached"], u["output"], u["reasoning"], latency, ok)
        (_current.get() or self.unscoped).add(call)
        return call

    def _trace_ctx(self, kind: str, req: dict):
        import contextlib
        if self.tracer is not None and self.tracer.on:
            return self.tracer.generation(kind, req.get("model"))
        return contextlib.nullcontext(None)

    def _trace_update(self, gen, call) -> None:
        if gen is None or call is None:
            return
        usage, cost = self.tracer.usage_and_cost(call, _price_for(call.model, self.pricing))
        try:
            gen.update(model=call.model, usage_details=usage, cost_details=cost,
                       metadata={"reasoning_tokens": call.reasoning, "http_ok": call.ok},
                       level="DEFAULT" if call.ok else "ERROR")
        except Exception:
            pass

    def install(self) -> "UsageMeter":
        """Patch httpx.Client.send / AsyncClient.send once, process-wide."""
        if self._installed:
            return self
        import httpx
        meter = self
        orig_send, orig_asend = httpx.Client.send, httpx.AsyncClient.send

        def send(client, request, *args, **kwargs):
            kind = meter._kind(request)
            if kind is None:
                return orig_send(client, request, *args, **kwargs)
            req = meter._req_json(request)
            with meter._trace_ctx(kind, req) as gen:
                t = time.perf_counter()
                try:
                    resp = orig_send(client, request, *args, **kwargs)
                except Exception:
                    meter._trace_update(gen, meter._record_http(kind, req, None, time.perf_counter() - t, False))
                    raise
                body = None
                if not req.get("stream"):
                    try:
                        resp.read()            # cached by httpx; the SDK can still parse it
                        body = resp.json()
                    except Exception:
                        pass
                call = meter._record_http(kind, req, body, time.perf_counter() - t, resp.status_code < 400)
                meter._trace_update(gen, call)
            return resp

        async def asend(client, request, *args, **kwargs):
            kind = meter._kind(request)
            if kind is None:
                return await orig_asend(client, request, *args, **kwargs)
            req = meter._req_json(request)
            with meter._trace_ctx(kind, req) as gen:
                t = time.perf_counter()
                try:
                    resp = await orig_asend(client, request, *args, **kwargs)
                except Exception:
                    meter._trace_update(gen, meter._record_http(kind, req, None, time.perf_counter() - t, False))
                    raise
                body = None
                if not req.get("stream"):
                    try:
                        await resp.aread()
                        body = resp.json()
                    except Exception:
                        pass
                call = meter._record_http(kind, req, body, time.perf_counter() - t, resp.status_code < 400)
                meter._trace_update(gen, call)
            return resp

        httpx.Client.send, httpx.AsyncClient.send = send, asend
        self._installed = True
        return self
