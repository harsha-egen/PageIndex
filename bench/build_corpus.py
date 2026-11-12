"""Step 1 — build the clinical corpus.

    python -m bench.build_corpus            # uses config.yaml scale (default: large)
    python -m bench.build_corpus --scale small
"""
from __future__ import annotations

import argparse
from collections import Counter

from bench.common import ROOT, append_manifest, load_config, load_manifest, manifest_path
from bench.data.fetch_labels import fetch_labels
from bench.data.fetch_trials import fetch_trials
from bench.data.synth_notes import generate_notes


def main() -> None:
    from bench.tracing import load_env
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["small", "medium", "large"])
    ap.add_argument("--only", choices=["trials", "labels", "notes"])
    args = ap.parse_args()
    cfg = load_config()
    if args.scale:
        cfg["scale"], cfg["scale_params"] = args.scale, cfg["scales"][args.scale]
    sp, base = cfg["scale_params"], ROOT / cfg["data"]["corpus_dir"]

    if args.only in (None, "trials"):
        print(f"Fetching {sp['trials']} clinical trial protocols ...")
        append_manifest(cfg, fetch_trials(cfg, base / "trials", sp["trials"]))
    if args.only in (None, "labels"):
        print(f"Fetching {sp['labels']} FDA labels ...")
        append_manifest(cfg, fetch_labels(cfg, base / "labels", sp["labels"]))
    if args.only in (None, "notes"):
        print(f"Generating {sp['notes']} synthetic notes ...")
        append_manifest(cfg, generate_notes(cfg, base / "notes", sp["notes"]))

    m = load_manifest(cfg)
    docs, pages = Counter(r["doc_type"] for r in m), Counter()
    for r in m:
        pages[r["doc_type"]] += r["pages"]
    print(f"\nCorpus at {manifest_path(cfg)}")
    for t in docs:
        print(f"  {t:16s} {docs[t]:5d} docs  {pages[t]:7d} pages")
    print(f"  {'TOTAL':16s} {sum(docs.values()):5d} docs  {sum(pages.values()):7d} pages")


if __name__ == "__main__":
    main()
