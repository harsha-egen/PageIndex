"""Offline checks for the side-by-side chat demo (no API keys, no network).

    python -m tests.test_chat_app
"""
from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from bench.chat_app import ChatDemo, make_handler
from bench.common import load_config
from bench.meter import Call, UsageMeter, tool_calls


def test_tool_calls_parsing():
    chat = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "get_page_content", "arguments": '{"pages": "4-5"}'}}]}}]}
    resp = {"output": [{"type": "function_call", "name": "get_tree", "arguments": "{}"}, {"type": "message"}]}
    assert tool_calls(chat) == [{"name": "get_page_content", "args": '{"pages": "4-5"}'}]
    assert tool_calls(resp) == [{"name": "get_tree", "args": "{}"}]
    assert tool_calls(None) == [] and tool_calls({"choices": [{"message": {}}]}) == []


def test_scope_on_call_hook():
    m = UsageMeter({"x": {"input": 1.0, "cached": 0.1, "output": 2.0}})
    seen = []
    with m.scope("t") as s:
        s.on_call = seen.append
        s.add(Call("chat", "x", 100, 0, 10, 0, 0.1))
    assert len(seen) == 1 and s.summary(m.pricing)["llm_calls"] == 1

    def boom(_):
        raise RuntimeError("listener failure must not break metering")
    with m.scope("t2") as s2:
        s2.on_call = boom
        s2.add(Call("chat", "x", 1, 0, 1, 0, 0.1))
    assert len(s2.calls) == 1


def _events(demo, question, doc_id):
    return list(demo.ask(question, doc_id))


def test_fake_demo_streams_both_pipelines():
    demo = ChatDemo(load_config(), fake=True)
    st = demo.status()
    assert st["fake"] and st["both_indexed"] == 2
    evs = _events(demo, "What is the starting dose?", "DEMO-LABEL-1")
    assert evs[-1] == {"type": "end"}
    for p in ("pageindex", "vector_rag"):
        mine = [e for e in evs if e.get("pipeline") == p]
        kinds = [e["type"] for e in mine]
        assert kinds[0] == "start" and kinds[-1] == "done", kinds
        calls = [e for e in mine if e["type"] == "call"]
        done = mine[-1]
        # live call events add up to the final summary
        assert len([c for c in calls if c["call"]["kind"] != "embeddings"]) == done["summary"]["llm_calls"]
        assert abs(sum(c["call"]["cost_usd"] for c in calls) - done["summary"]["cost_usd"]) < 1e-5
        assert done["summary"]["cost_usd"] > 0 and done["answer"]
    pi_calls = [e for e in evs if e.get("pipeline") == "pageindex" and e["type"] == "call"]
    assert len(pi_calls) == 4 and pi_calls[1]["call"]["tools"][0]["name"] == "get_page_content"
    # an un-indexed document is a per-pipeline error, not a crash
    evs = _events(demo, "q", "NOT-INDEXED")
    assert [e["type"] for e in evs if e.get("pipeline")] == ["error", "error"]


def test_http_endpoints():
    demo = ChatDemo(load_config(), fake=True)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(demo))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"PageIndex vs Vector RAG" in urllib.request.urlopen(base + "/").read()
        st = json.load(urllib.request.urlopen(base + "/api/status"))
        assert len(st["docs"]) == 2
        req = urllib.request.Request(base + "/api/ask", data=json.dumps({"question": "dose?", "doc_id": None}).encode(),
                                     headers={"Content-Type": "application/json"})
        lines = [json.loads(l) for l in urllib.request.urlopen(req, timeout=60).read().splitlines() if l.strip()]
        assert lines[-1]["type"] == "end" and sum(1 for e in lines if e["type"] == "done") == 2
    finally:
        srv.shutdown()


if __name__ == "__main__":
    for fn in (test_tool_calls_parsing, test_scope_on_call_hook, test_fake_demo_streams_both_pipelines,
               test_http_endpoints):
        fn()
        print("PASS", fn.__name__)
