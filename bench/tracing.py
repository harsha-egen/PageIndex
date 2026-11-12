"""Langfuse tracing for the benchmark (Langfuse Python SDK v3+ / OpenTelemetry based).

What lands in Langfuse
----------------------
* one TRACE per query      name "<pipeline>/<scope>"         e.g. "pageindex/document"
* one TRACE per indexed doc name "<pipeline>/index"
* one GENERATION (or EMBEDDING) per OpenAI HTTP call inside it, emitted by bench/meter.py,
  so PageIndex's internal agent turns (LiteLLM -> httpx) appear as separate child calls,
  with exact token buckets and USD cost computed from config.yaml prices.
* SCORES on each query trace: accuracy (LLM judge), accuracy/<q_type>, page_hit, wrong_doc,
  latency_s, llm_calls.

Dimensions for dashboards
-------------------------
user_id    = "<pipeline>/<scope>"  -> the "User" breakdown compares pipelines in every view
session_id = run name              -> one session per benchmark run
tags       = pipeline, scope, doc_type, q_type, "query"/"index", run type
environment= "benchmark" (measured) or "modeled" (replayed preview) -> dashboard env filter

Credentials come from the environment or a local .env file (never commit it):
LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL (or LANGFUSE_HOST).
"""
from __future__ import annotations

import contextlib
import os
import warnings

from bench.common import ROOT


def load_env(path=ROOT / ".env") -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    # SDK versions differ in which name they read; set both.
    if os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]
    if os.getenv("LANGFUSE_HOST") and not os.getenv("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_BASE_URL"] = os.environ["LANGFUSE_HOST"]


def _s(v) -> str:
    return "" if v is None else str(v)[:200]


class Tracer:
    def __init__(self, cfg: dict | None = None, environment: str = "benchmark", run_name: str | None = None):
        load_env()
        lcfg = (cfg or {}).get("langfuse", {})
        self.run_name = run_name or (cfg or {}).get("run_name", "clinical-bench")
        self.lf = None
        if not lcfg.get("enabled", True):
            return
        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            print("[langfuse] keys not set -> tracing off (metrics still saved to results/)")
            return
        os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", environment)
        try:
            from langfuse import get_client
            self.lf = get_client()
            if hasattr(self.lf, "auth_check") and not self.lf.auth_check():
                print("[langfuse] auth_check failed: check keys and LANGFUSE_BASE_URL -> tracing off")
                self.lf = None
            else:
                print(f"[langfuse] tracing on -> {os.getenv('LANGFUSE_BASE_URL')} env={environment} session={self.run_name}")
        except ImportError:
            print("[langfuse] package not installed (pip install langfuse) -> tracing off")
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"[langfuse] disabled: {e}")
            self.lf = None

    @property
    def on(self) -> bool:
        return self.lf is not None

    # ------------------------------------------------------------------ traces
    @contextlib.contextmanager
    def trace(self, name: str, group: str, tags: list[str], metadata: dict, input=None):
        """Root span of one query / one indexed document. Yields the span (or None)."""
        if not self.on:
            yield None
            return
        from langfuse import propagate_attributes
        meta = {k: _s(v) for k, v in metadata.items()}
        with propagate_attributes(user_id=group, session_id=self.run_name, tags=[t for t in tags if t],
                                  metadata=meta, trace_name=name):
            with self.lf.start_as_current_observation(as_type="span", name=name, input=input) as span:
                yield span

    def current_trace_id(self) -> str | None:
        if not self.on:
            return None
        try:
            return self.lf.get_current_trace_id()
        except Exception:
            return None

    # ------------------------------------------------------------- generations
    @contextlib.contextmanager
    def generation(self, kind: str, model: str | None):
        """Child observation around one OpenAI HTTP call (used by bench/meter.py)."""
        if not self.on:
            yield None
            return
        as_type = "embedding" if kind == "embeddings" else "generation"
        with self.lf.start_as_current_observation(as_type=as_type, name=f"openai.{kind}", model=model) as g:
            yield g

    @staticmethod
    def usage_and_cost(call, price: dict) -> tuple[dict, dict]:
        """Langfuse wants EXCLUSIVE buckets: uncached input, cached input, output."""
        uncached = max(call.input - call.cached, 0)
        usage = {"input": uncached, "input_cached_tokens": call.cached, "output": call.output}
        cost = {"input": uncached * price.get("input", 0) / 1e6,
                "input_cached_tokens": call.cached * price.get("cached", price.get("input", 0)) / 1e6,
                "output": call.output * price.get("output", 0) / 1e6}
        return usage, cost

    # ------------------------------------------------------------------ scores
    def score(self, trace_id: str | None, name: str, value, comment: str | None = None) -> None:
        if not (self.on and trace_id) or value is None:
            return
        try:
            self.lf.create_score(trace_id=trace_id, name=name, value=float(value),
                                 data_type="NUMERIC", comment=comment)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"[langfuse] score {name} failed: {e}")

    def flush(self) -> None:
        if self.on:
            self.lf.flush()
