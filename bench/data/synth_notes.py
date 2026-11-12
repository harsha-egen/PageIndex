"""Short synthetic clinical notes (1-2 pages each) — the *flat* side of the corpus.

Two sources:
  1. Synthea FHIR R4 bundles (set data.synthea_dir). Generate them with:
       java -jar synthea-with-dependencies.jar -p 1200 --exporter.fhir.export true
  2. Built-in generator (default) — no Java needed, deterministic from the seed.

No real patient data is used anywhere. Every note is fully synthetic.
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

from bench.common import read_pdf_pages
from bench.data.pdf_render import render_pdf

FIRST = ["James", "Maria", "Robert", "Linda", "Wei", "Aisha", "Carlos", "Priya", "John", "Fatima",
         "Michael", "Elena", "David", "Mei", "Samuel", "Grace", "Ahmed", "Olivia", "Daniel", "Sofia"]
LAST = ["Smith", "Garcia", "Nguyen", "Patel", "Johnson", "Kim", "Martinez", "Brown", "Okafor", "Chen",
        "Lopez", "Wilson", "Singh", "Davis", "Hernandez", "Ali", "Taylor", "Moore", "Clark", "Lee"]
PROBLEMS = {
    "Type 2 diabetes mellitus": {"meds": ["metformin 1000 mg PO BID", "empagliflozin 10 mg PO daily",
                                          "insulin glargine 18 units SC nightly"],
                                 "labs": [("HbA1c", 6.4, 10.8, "%"), ("Fasting glucose", 95, 240, "mg/dL")]},
    "Essential hypertension": {"meds": ["lisinopril 20 mg PO daily", "amlodipine 5 mg PO daily",
                                        "hydrochlorothiazide 25 mg PO daily"], "labs": []},
    "Heart failure with reduced ejection fraction": {
        "meds": ["sacubitril/valsartan 49/51 mg PO BID", "carvedilol 12.5 mg PO BID", "spironolactone 25 mg PO daily"],
        "labs": [("NT-proBNP", 300, 6500, "pg/mL"), ("LVEF", 20, 40, "%")]},
    "Chronic kidney disease stage 3": {"meds": ["dapagliflozin 10 mg PO daily"],
                                       "labs": [("eGFR", 30, 59, "mL/min/1.73m2"), ("Creatinine", 1.3, 2.4, "mg/dL")]},
    "Hyperlipidemia": {"meds": ["atorvastatin 40 mg PO nightly", "rosuvastatin 20 mg PO daily"],
                       "labs": [("LDL cholesterol", 60, 190, "mg/dL")]},
    "Atrial fibrillation": {"meds": ["apixaban 5 mg PO BID", "metoprolol succinate 50 mg PO daily"],
                            "labs": [("CHA2DS2-VASc score", 2, 6, "")]},
    "Asthma": {"meds": ["budesonide/formoterol 160/4.5 mcg 2 puffs BID", "albuterol HFA PRN"],
               "labs": [("FEV1 % predicted", 55, 92, "%")]},
    "Major depressive disorder": {"meds": ["sertraline 100 mg PO daily", "bupropion XL 150 mg PO daily"],
                                  "labs": [("PHQ-9", 4, 22, "")]},
    "Rheumatoid arthritis": {"meds": ["methotrexate 15 mg PO weekly", "folic acid 1 mg PO daily"],
                             "labs": [("CRP", 0.5, 4.8, "mg/dL")]},
    "Hypothyroidism": {"meds": ["levothyroxine 88 mcg PO daily"], "labs": [("TSH", 0.6, 9.5, "mIU/L")]},
}
ALLERGIES = ["NKDA", "penicillin (rash)", "sulfa drugs (hives)", "codeine (nausea)", "iodinated contrast (urticaria)"]
NOTE_TYPES = ["Outpatient Progress Note", "Discharge Summary", "Specialty Consult Note"]


def _note_sections(rng: random.Random, mrn: str, name: str, age: int, sex: str, when: date) -> tuple[list[dict], dict]:
    probs = rng.sample(list(PROBLEMS), k=rng.randint(2, 4))
    meds, labs = [], []
    for p in probs:
        meds += rng.sample(PROBLEMS[p]["meds"], k=min(len(PROBLEMS[p]["meds"]), rng.randint(1, 2)))
        for lab, lo, hi, unit in PROBLEMS[p]["labs"]:
            v = round(rng.uniform(lo, hi), 1)
            labs.append(f"{lab}: {v} {unit}".strip())
    bp = f"{rng.randint(112, 168)}/{rng.randint(68, 98)}"
    vitals = f"BP {bp} mmHg, HR {rng.randint(58, 104)} bpm, Temp {round(rng.uniform(36.4, 37.6), 1)} C, " \
             f"SpO2 {rng.randint(92, 99)}% on room air, BMI {round(rng.uniform(21, 38), 1)}"
    allergy = rng.choice(ALLERGIES)
    plan_item = rng.choice(probs)
    change = rng.choice(["increase", "start", "hold", "continue"])
    plan = (f"{plan_item}: {change} current regimen; repeat labs in {rng.choice([4, 6, 8, 12])} weeks. "
            f"Follow up in clinic in {rng.choice([2, 4, 6, 12])} weeks. Patient counseled on diet, adherence and warning signs.")
    sections = [
        {"title": "Patient Information", "text": f"Name: {name}\nMRN: {mrn}\nAge/Sex: {age} {sex}\nDate of service: {when.isoformat()}"},
        {"title": "Chief Complaint / Reason for Visit", "text": f"Follow-up of {', '.join(probs[:2]).lower()}."},
        {"title": "Problem List", "text": "\n".join(f"- {p}" for p in probs)},
        {"title": "Medications", "text": "\n".join(f"- {m}" for m in meds)},
        {"title": "Allergies", "text": allergy},
        {"title": "Vital Signs", "text": vitals},
        {"title": "Laboratory Results", "text": "\n".join(f"- {l}" for l in labs) or "No new labs."},
        {"title": "Assessment and Plan", "text": plan},
    ]
    facts = {"problems": probs, "meds": meds, "labs": labs, "allergy": allergy, "bp": bp, "plan": plan}
    return sections, facts


def generate_notes(cfg: dict, out_dir: Path, n: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg["data"].get("synthea_dir"):
        return from_synthea(Path(cfg["data"]["synthea_dir"]), out_dir, n)
    rng = random.Random(cfg["seed"])
    rows = []
    for i in range(n):
        mrn = f"{100000 + i * 7 + rng.randint(0, 6)}"
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        age, sex = rng.randint(28, 89), rng.choice(["M", "F"])
        when = date(2025, 1, 1) + timedelta(days=rng.randint(0, 600))
        ntype = rng.choice(NOTE_TYPES)
        sections, facts = _note_sections(rng, mrn, name, age, sex, when)
        doc_id = f"NOTE-{mrn}"
        path = out_dir / f"{doc_id}.pdf"
        render_pdf(path, f"{ntype} — {name} (MRN {mrn})", sections)
        rows.append({"doc_id": doc_id, "doc_type": "clinical_note", "title": f"{ntype} — MRN {mrn}",
                     "patient": name, "mrn": mrn, "path": str(path), "pages": len(read_pdf_pages(path)),
                     "facts": facts, "source": "synthetic (built-in generator)"})
    print(f"  generated {n} synthetic notes")
    return rows


def from_synthea(fhir_dir: Path, out_dir: Path, n: int) -> list[dict]:
    """Convert Synthea FHIR R4 patient bundles into short notes."""
    rows = []
    for fp in sorted(fhir_dir.glob("*.json"))[: n * 2]:
        if len(rows) >= n or fp.name.startswith(("hospital", "practitioner")):
            continue
        bundle = json.loads(fp.read_text())
        res = [e["resource"] for e in bundle.get("entry", [])]
        pat = next((r for r in res if r["resourceType"] == "Patient"), None)
        if not pat:
            continue
        nm = pat.get("name", [{}])[0]
        name = " ".join(nm.get("given", [])[:1] + [nm.get("family", "")])
        mrn = pat["id"][:8]
        conds = sorted({r["code"]["text"] for r in res if r["resourceType"] == "Condition"
                        and r.get("clinicalStatus", {}).get("coding", [{}])[0].get("code") == "active"
                        and "text" in r.get("code", {})})[:8]
        meds = sorted({r.get("medicationCodeableConcept", {}).get("text", "") for r in res
                       if r["resourceType"] == "MedicationRequest" and r.get("status") == "active"} - {""})[:10]
        obs = [r for r in res if r["resourceType"] == "Observation" and "valueQuantity" in r]
        obs = sorted(obs, key=lambda r: r.get("effectiveDateTime", ""))[-12:]
        labs = [f"{o['code'].get('text', '?')}: {round(o['valueQuantity']['value'], 1)} "
                f"{o['valueQuantity'].get('unit', '')} ({o.get('effectiveDateTime', '')[:10]})" for o in obs]
        allergies = [r.get("code", {}).get("text", "") for r in res if r["resourceType"] == "AllergyIntolerance"]
        sections = [
            {"title": "Patient Information", "text": f"Name: {name}\nMRN: {mrn}\nBirth date: {pat.get('birthDate')}\nSex: {pat.get('gender')}"},
            {"title": "Active Problems", "text": "\n".join(f"- {c}" for c in conds) or "None recorded."},
            {"title": "Active Medications", "text": "\n".join(f"- {m}" for m in meds) or "None."},
            {"title": "Allergies", "text": ", ".join(allergies) or "NKDA"},
            {"title": "Recent Observations", "text": "\n".join(f"- {l}" for l in labs) or "None."},
        ]
        doc_id = f"NOTE-{mrn}"
        path = out_dir / f"{doc_id}.pdf"
        render_pdf(path, f"Patient Summary — {name} (MRN {mrn})", sections)
        rows.append({"doc_id": doc_id, "doc_type": "clinical_note", "title": f"Patient Summary — MRN {mrn}",
                     "patient": name, "mrn": mrn, "path": str(path), "pages": len(read_pdf_pages(path)),
                     "facts": {"problems": conds, "meds": meds, "labs": labs}, "source": "Synthea FHIR R4"})
    print(f"  converted {len(rows)} Synthea patients")
    return rows
