"""Offline test of the Langfuse integration with a stand-in SDK (no network, no keys)."""
import contextlib, json, os, sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVENTS = []

class Obs:
    def __init__(self, kind, name, **kw): self.kind, self.name, self.kw, self.updates = kind, name, kw, []
    def update(self, **kw): self.updates.append(kw)

class FakeClient:
    stack = []
    def auth_check(self): return True
    @contextlib.contextmanager
    def start_as_current_observation(self, as_type, name, **kw):
        o = Obs(as_type, name, **kw); o.parent = self.stack[-1].name if self.stack else None
        o.trace = self.stack[0].trace if self.stack else f"t{len(EVENTS)}"
        EVENTS.append(o); self.stack.append(o)
        try: yield o
        finally: self.stack.pop()
    def get_current_trace_id(self): return self.stack[0].trace if self.stack else None
    def create_score(self, **kw): EVENTS.append(("score", kw))
    def flush(self): EVENTS.append("flush")

ATTRS = []
@contextlib.contextmanager
def propagate_attributes(**kw):
    ATTRS.append(kw); yield

fake = types.ModuleType("langfuse")
fake.get_client = lambda: FakeClient()
fake.propagate_attributes = propagate_attributes
sys.modules["langfuse"] = fake
os.environ.update(LANGFUSE_PUBLIC_KEY="pk-test", LANGFUSE_SECRET_KEY="sk-test", LANGFUSE_BASE_URL="https://example")

from tests.test_offline import _fake_httpx  # noqa: E402
try:
    import httpx  # noqa: F401
except ImportError:
    httpx = _fake_httpx()
import httpx  # noqa: E402,F811
from bench.meter import UsageMeter  # noqa: E402
from bench.tracing import Tracer  # noqa: E402


def test_meter_emits_generations_inside_trace():
    pricing = {"gpt-5.6-terra": {"input": 2.0, "cached": 0.2, "output": 12.0}}
    meter = UsageMeter(pricing).install()
    tracer = Tracer({"langfuse": {"enabled": True}}, environment="benchmark")
    assert tracer.on
    meter.tracer = tracer
    def handler(request):
        return httpx.Response(200, json={"model": "gpt-5.6-terra", "usage": {
            "prompt_tokens": 1000, "completion_tokens": 200, "prompt_tokens_details": {"cached_tokens": 400}}})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    EVENTS.clear()
    with tracer.trace("pageindex/document", "pageindex/document", ["pageindex", "document", "query"], {"q_id": "q1"}, input="Q?"), \
            meter.scope("x") as s:
        for _ in range(3):
            client.post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-5.6-terra"})
        tid = tracer.current_trace_id()
    tracer.score(tid, "accuracy", 1)
    gens = [e for e in EVENTS if isinstance(e, Obs) and e.kind == "generation"]
    assert len(gens) == 3 and all(g.parent == "pageindex/document" for g in gens)
    u = gens[0].updates[-1]
    assert u["usage_details"] == {"input": 600, "input_cached_tokens": 400, "output": 200}
    assert abs(sum(u["cost_details"].values()) - (600 * 2 + 400 * 0.2 + 200 * 12) / 1e6) < 1e-12
    assert ATTRS[-1]["user_id"] == "pageindex/document" and "query" in ATTRS[-1]["tags"]
    assert ("score", {"trace_id": tid, "name": "accuracy", "value": 1.0, "data_type": "NUMERIC", "comment": None}) in EVENTS
    assert s.summary(pricing)["llm_calls"] == 3


def test_replay_modeled_sample():
    import bench.langfuse_replay as rp
    EVENTS.clear()
    sys.argv = ["x", "--dir", str(Path(__file__).resolve().parent.parent / "results" / "modeled"), "--limit", "60"]
    rp.main()
    traces = [e for e in EVENTS if isinstance(e, Obs) and e.kind == "span"]
    scores = [e for e in EVENTS if isinstance(e, tuple)]
    names = {t.name for t in traces}
    assert names == {"vector_rag/document", "vector_rag/corpus", "pageindex/document", "pageindex/corpus"}
    assert {s[1]["name"] for s in scores} >= {"accuracy", "page_hit", "latency_s", "llm_calls"}
    assert EVENTS[-1] == "flush"


if __name__ == "__main__":
    test_meter_emits_generations_inside_trace(); print("PASS meter -> langfuse generations")
    test_replay_modeled_sample(); print("PASS modeled replay")
