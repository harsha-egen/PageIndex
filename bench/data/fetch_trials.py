"""Download full clinical trial protocol PDFs from ClinicalTrials.gov (API v2).

Protocols are the ideal PageIndex showcase: 50-300 pages, deep numbered sections
(Eligibility 5.1/5.2, Dose modification 6.4, Schedule of Assessments tables, AE grading),
and questions that need cross-references between sections.

Public data, no key needed. Documents are stored at
https://cdn.clinicaltrials.gov/large-docs/<last 2 digits of NCT>/<NCT>/<filename>
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

from bench.common import read_pdf_pages

API = "https://clinicaltrials.gov/api/v2/studies"
CDN = "https://cdn.clinicaltrials.gov/large-docs/{tail}/{nct}/{fname}"


def _candidates(term: str, page_size: int = 100, max_pages: int = 3):
    token = None
    for _ in range(max_pages):
        params = {
            "query.cond": term,
            "aggFilters": "docs:prot",          # studies that posted a protocol document
            "filter.overallStatus": "COMPLETED",
            "fields": "NCTId,BriefTitle,Phase,LargeDocumentModule",
            "pageSize": page_size,
        }
        if token:
            params["pageToken"] = token
        r = requests.get(API, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        for st in js.get("studies", []):
            ps = st.get("protocolSection", {})
            nct = ps.get("identificationModule", {}).get("nctId")
            title = ps.get("identificationModule", {}).get("briefTitle", "")
            docs = st.get("documentSection", {}).get("largeDocumentModule", {}).get("largeDocs", [])
            for d in docs:
                # Prefer the protocol-only file; a combined Prot_SAP also works.
                if d.get("hasProtocol") and d.get("filename", "").lower().endswith(".pdf"):
                    yield nct, title, d
                    break
        token = js.get("nextPageToken")
        if not token:
            break


def fetch_trials(cfg: dict, out_dir: Path, n: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    min_pages = cfg["data"]["trial_min_pages"]
    terms = cfg["data"]["trials_condition_terms"]
    per_term = max(1, -(-n // len(terms)))
    rows: list[dict] = []
    for term in terms:
        got = 0
        for nct, title, d in _candidates(term):
            if got >= per_term or len(rows) >= n:
                break
            if d.get("size") and d["size"] > 25_000_000:      # skip giant scans
                continue
            path = out_dir / f"{nct}.pdf"
            if not path.exists():
                url = CDN.format(tail=nct[-2:], nct=nct, fname=d["filename"])
                try:
                    resp = requests.get(url, timeout=120)
                    resp.raise_for_status()
                    path.write_bytes(resp.content)
                except Exception as e:  # noqa: BLE001
                    print(f"  skip {nct}: {e}")
                    continue
                time.sleep(0.3)  # be polite to the CDN
            try:
                pages = read_pdf_pages(path)
            except Exception:
                path.unlink(missing_ok=True)
                continue
            text_chars = sum(len(p) for p in pages)
            # Protocols must be long AND text-based (scanned PDFs need PageIndex Cloud OCR)
            if len(pages) < min_pages or text_chars < len(pages) * 400:
                path.unlink(missing_ok=True)
                continue
            rows.append({"doc_id": nct, "doc_type": "trial_protocol", "title": title,
                         "condition": term, "path": str(path), "pages": len(pages),
                         "chars": text_chars, "source": "ClinicalTrials.gov"})
            got += 1
            print(f"  trial {nct}: {len(pages)} pages — {title[:70]}")
    return rows
