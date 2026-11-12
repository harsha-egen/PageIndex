"""PageIndex pipeline: tree index per document + agentic LLM tree search.

Uses the PageIndex SDK (pip install -U pageindex), local mode by default:
  index -> PageIndexClient.submit_document(pdf)      (LLM summarises/refines the tree)
  query -> PageIndexClient.chat(q, doc_id=..., citations=True, protocol="chat_completions")

Corpus scope ("which trial says X?") calls chat() WITHOUT doc_id, which makes PageIndex
search the whole local library — the harder, more realistic case at volume.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from bench.pipelines.vector_rag import ANSWER_SYSTEM

INSTRUCTIONS = ANSWER_SYSTEM.replace("Cite every claim as [doc_id p.N]. ", "")


class PageIndexRAG:
    name = "pageindex"

    def __init__(self, cfg: dict, store_dir: Path):
        from pageindex import PageIndexClient  # imported lazily so the rest runs without it
        self.cfg, self.dir = cfg, store_dir
        pcfg, m = cfg["pageindex"], cfg["models"]
        store_dir.mkdir(parents=True, exist_ok=True)
        if pcfg["mode"] == "cloud":
            index = "cloud"
        else:
            index = {"model": m["pageindex_index"], "storage_path": str(store_dir / "library")}
        self.client = PageIndexClient(index=index, chat=m["chat"], instructions=INSTRUCTIONS)
        self.map_path = store_dir / "doc_map.json"
        self.doc_map: dict[str, str] = json.loads(self.map_path.read_text()) if self.map_path.exists() else {}
        self.rev = {v: k for k, v in self.doc_map.items()}

    # ------------------------------------------------------------ indexing
    def _tree_stats(self, pi_id: str) -> dict:
        """Node count / depth of the generated tree (best effort across SDK versions)."""
        for fn in ("get_document_structure", "get_tree", "get_structure"):
            f = getattr(self.client, fn, None)
            if not f:
                continue
            try:
                tree = f(pi_id)
                tree = tree.get("structure", tree) if isinstance(tree, dict) else tree
                nodes, depth = 0, 0

                def walk(n, d):
                    nonlocal nodes, depth
                    for x in (n if isinstance(n, list) else [n]):
                        if isinstance(x, dict):
                            nodes += 1
                            depth = max(depth, d)
                            walk(x.get("nodes") or x.get("children") or [], d + 1)
                walk(tree, 1)
                return {"tree_nodes": nodes, "tree_depth": depth,
                        "tree_json_chars": len(json.dumps(tree, default=str))}
            except Exception:
                continue
        return {"tree_nodes": None, "tree_depth": None, "tree_json_chars": None}

    def index_document(self, doc: dict) -> dict:
        if doc["doc_id"] in self.doc_map:
            return {"skipped": True}
        t0 = time.perf_counter()
        res = self.client.submit_document(doc["path"], wait=True)
        pi_id = res["doc_id"]
        t1 = time.perf_counter()
        self.doc_map[doc["doc_id"]] = pi_id
        self.rev[pi_id] = doc["doc_id"]
        self.map_path.write_text(json.dumps(self.doc_map, indent=1))
        lib = self.dir / "library"
        storage = sum(f.stat().st_size for f in lib.rglob("*") if f.is_file() and pi_id in str(f)) if lib.exists() else None
        return {"index_s": t1 - t0, "storage_bytes": storage, **self._tree_stats(pi_id)}

    def finalize_index(self) -> None:
        pass

    # ------------------------------------------------------------ querying
    def _resolve_doc(self, c: dict) -> str:
        if c.get("doc_id") in self.rev:
            return self.rev[c["doc_id"]]
        return Path(str(c.get("document", "?"))).stem

    def query(self, question: str, doc_id: str | None) -> dict:
        kwargs = {"citations": self.cfg["pageindex"]["citations"], "protocol": "chat_completions",
                  "max_turns": self.cfg["pageindex"]["max_turns"]}
        if self.cfg["models"].get("reasoning_effort"):
            kwargs["reasoning_effort"] = self.cfg["models"]["reasoning_effort"]
        if doc_id:
            kwargs["doc_id"] = self.doc_map[doc_id]
        t0 = time.perf_counter()
        resp = self.client.chat(question, **kwargs)
        t1 = time.perf_counter()
        answer = resp["choices"][0]["message"]["content"] or ""
        try:
            cites = self.client.get_citations(answer)
        except Exception:
            cites = []
        retrieved = sorted({(self._resolve_doc(c), int(c["page"])) for c in cites if c.get("page")})
        try:
            clean_answer = self.client.resolve_citations(answer)["answer"]
        except Exception:
            clean_answer = answer
        return {"answer": clean_answer,
                "retrieved": [{"doc_id": d, "page": p} for d, p in retrieved],
                "protocol_usage": resp.get("usage") or {},
                "stage_s": {"agent_total": t1 - t0},
                "turns": None}  # filled from the meter (number of chat calls)
