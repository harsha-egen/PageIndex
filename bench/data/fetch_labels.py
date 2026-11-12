"""FDA prescribing information (drug labels) from openFDA, rendered to PDF.

Labels are medium-length (15-60 pages), follow a fixed numbered structure
(1 Indications ... 17 Patient Counseling) and are full of cross-references
("see Warnings and Precautions (5.3)"), tables and population-specific dosing.
Public data; an optional OPENFDA_API_KEY raises rate limits.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import requests

from bench.common import read_pdf_pages
from bench.data.pdf_render import render_pdf

API = "https://api.fda.gov/drug/label.json"

# Standard PLR label order, numbered as on a real label.
SECTIONS = [
    ("boxed_warning", "BOXED WARNING"),
    ("indications_and_usage", "1 INDICATIONS AND USAGE"),
    ("dosage_and_administration", "2 DOSAGE AND ADMINISTRATION"),
    ("dosage_forms_and_strengths", "3 DOSAGE FORMS AND STRENGTHS"),
    ("contraindications", "4 CONTRAINDICATIONS"),
    ("warnings_and_cautions", "5 WARNINGS AND PRECAUTIONS"),
    ("adverse_reactions", "6 ADVERSE REACTIONS"),
    ("drug_interactions", "7 DRUG INTERACTIONS"),
    ("use_in_specific_populations", "8 USE IN SPECIFIC POPULATIONS"),
    ("overdosage", "10 OVERDOSAGE"),
    ("description", "11 DESCRIPTION"),
    ("clinical_pharmacology", "12 CLINICAL PHARMACOLOGY"),
    ("nonclinical_toxicology", "13 NONCLINICAL TOXICOLOGY"),
    ("clinical_studies", "14 CLINICAL STUDIES"),
    ("how_supplied", "16 HOW SUPPLIED/STORAGE AND HANDLING"),
    ("patient_counseling_information", "17 PATIENT COUNSELING INFORMATION"),
]
SUB = re.compile(r"(?<=\s)(\d{1,2}\.\d{1,2})\s+([A-Z][A-Za-z ,/\-()]{3,80}?)(?=\s{1,3}[A-Z][a-z])")


def _split_subsections(text: str) -> tuple[str, list[dict]]:
    """openFDA flattens '5.1 Hepatotoxicity ...' into one string; split it back out."""
    marks = list(SUB.finditer(" " + text))
    if len(marks) < 2:
        return text, []
    intro = text[: max(marks[0].start() - 1, 0)]
    children = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() - 1 if i + 1 < len(marks) else len(text)
        body = text[m.end() - 1: end].strip()
        children.append({"title": f"{m.group(1)} {m.group(2).strip()}", "text": body})
    return intro, children


def _paragraphs(text: str) -> str:
    # break very long runs into paragraphs at sentence boundaries (~700 chars)
    out, buf = [], ""
    for sent in re.split(r"(?<=[.;])\s+", text):
        buf += sent + " "
        if len(buf) > 700:
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return "\n".join(out)


def fetch_labels(cfg: dict, out_dir: Path, n: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    key = os.getenv("OPENFDA_API_KEY")
    min_chars = cfg["data"]["label_min_chars"]
    rows, seen_names, skip = [], set(), 0
    while len(rows) < n and skip < 5000:
        params = {"search": 'openfda.product_type:"HUMAN PRESCRIPTION DRUG" AND _exists_:clinical_studies',
                  "limit": 100, "skip": skip}
        if key:
            params["api_key"] = key
        r = requests.get(API, params=params, timeout=60)
        r.raise_for_status()
        skip += 100
        for lab in r.json().get("results", []):
            if len(rows) >= n:
                break
            name = (lab.get("openfda", {}).get("generic_name") or ["?"])[0].lower()
            brand = (lab.get("openfda", {}).get("brand_name") or [name])[0]
            if name in seen_names:
                continue
            total = sum(len(" ".join(lab.get(k, []))) for k, _ in SECTIONS)
            if total < min_chars:
                continue
            seen_names.add(name)
            doc_id = "LBL-" + lab["set_id"][:8]
            secs = []
            for k, title in SECTIONS:
                if k not in lab:
                    continue
                raw = " ".join(lab[k])
                raw = re.sub(r"^\s*(\d+\s+)?[A-Z /&]{4,}\s+", "", raw, count=1)  # drop dup heading
                intro, kids = _split_subsections(raw)
                secs.append({"title": title, "text": _paragraphs(intro),
                             "children": [{"title": c["title"], "text": _paragraphs(c["text"])} for c in kids]})
            path = out_dir / f"{doc_id}.pdf"
            title = f"{brand} ({name}) — Full Prescribing Information"
            render_pdf(path, title, secs)
            pages = len(read_pdf_pages(path))
            rows.append({"doc_id": doc_id, "doc_type": "fda_label", "title": title,
                         "drug": name, "brand": brand, "set_id": lab["set_id"],
                         "path": str(path), "pages": pages, "chars": total, "source": "openFDA"})
            print(f"  label {doc_id}: {pages} pages — {brand}")
        time.sleep(0.25)
    return rows
