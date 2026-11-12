"""Shared helpers: config, PDF reading, token counting, corpus manifest."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["scale_params"] = cfg["scales"][cfg["scale"]]
    return cfg


# ----------------------------------------------------------------- tokens
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text, disallowed_special=()))

    def encode(text: str) -> list[int]:
        return _ENC.encode(text, disallowed_special=())

    def decode(tokens: list[int]) -> str:
        return _ENC.decode(tokens)
except Exception:  # offline fallback: ~4 chars per token
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def encode(text: str) -> list[int]:  # type: ignore[misc]
        return list(range(0, len(text), 4))

    def decode(tokens: list[int]) -> str:  # not used in fallback mode
        raise RuntimeError("tiktoken not installed")


# ----------------------------------------------------------------- PDFs
def read_pdf_pages(path: str | os.PathLike) -> list[str]:
    """Return text per page (1 entry per page, page 1 = index 0)."""
    try:
        import fitz  # PyMuPDF
        with fitz.open(path) as doc:
            return [p.get_text("text") for p in doc]
    except ImportError:
        from pypdf import PdfReader
        return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]


def read_pdf_toc(path: str | os.PathLike) -> list[tuple[int, str, int]]:
    """(level, title, page) from PDF bookmarks, if any."""
    try:
        import fitz
        with fitz.open(path) as doc:
            return [(lvl, t, pg) for lvl, t, pg in doc.get_toc()]
    except ImportError:
        from pypdf import PdfReader
        r = PdfReader(str(path))
        out: list[tuple[int, str, int]] = []

        def walk(items, lvl):
            for it in items:
                if isinstance(it, list):
                    walk(it, lvl + 1)
                else:
                    try:
                        out.append((lvl, it.title, r.get_destination_page_number(it) + 1))
                    except Exception:
                        pass
        walk(r.outline, 1)
        return out


def clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


# ----------------------------------------------------------------- manifest
def manifest_path(cfg: dict) -> Path:
    return ROOT / cfg["data"]["corpus_dir"] / "manifest.jsonl"


def load_manifest(cfg: dict) -> list[dict]:
    p = manifest_path(cfg)
    if not p.exists():
        raise SystemExit(f"No corpus manifest at {p}. Run: python -m bench.build_corpus")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def append_manifest(cfg: dict, rows: list[dict]) -> None:
    p = manifest_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = {r["doc_id"] for r in load_manifest(cfg)} if p.exists() else set()
    with open(p, "a") as f:
        for r in rows:
            if r["doc_id"] not in existing:
                f.write(json.dumps(r) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
