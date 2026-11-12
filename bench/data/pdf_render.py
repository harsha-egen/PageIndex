"""Render structured text (title + nested sections) into a PDF with real headings and
PDF bookmarks, so both pipelines read the same file the way they would read a real
label or note."""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

_ss = getSampleStyleSheet()
STYLES = {
    0: ParagraphStyle("T", parent=_ss["Title"], fontSize=15, spaceAfter=10),
    1: ParagraphStyle("H1", parent=_ss["Heading1"], fontSize=12.5, spaceBefore=10, spaceAfter=4),
    2: ParagraphStyle("H2", parent=_ss["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3),
    "body": ParagraphStyle("B", parent=_ss["BodyText"], fontSize=9.5, leading=12.5),
}


class _Doc(SimpleDocTemplate):
    def afterFlowable(self, fl):  # add a bookmark for every heading
        lvl = getattr(fl, "_outline_level", None)
        if lvl is None:
            return
        key = f"h{id(fl)}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(fl._outline_text[:120], key, level=lvl, closed=True)


def _heading(text: str, level: int) -> Paragraph:
    p = Paragraph(escape(text), STYLES[min(level + 1, 2)])
    p._outline_level = level
    p._outline_text = text
    return p


def render_pdf(path: Path, title: str, sections: list[dict]) -> None:
    """sections: [{"title": str, "text": str, "children": [same shape]}]"""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _Doc(str(path), pagesize=LETTER, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
               topMargin=0.7 * inch, bottomMargin=0.7 * inch, title=title)
    story = [Paragraph(escape(title), STYLES[0])]

    def emit(secs, level):
        for s in secs:
            story.append(_heading(s["title"], level))
            for para in [p for p in s.get("text", "").split("\n") if p.strip()]:
                story.append(Paragraph(escape(para.strip()), STYLES["body"]))
                story.append(Spacer(1, 3))
            emit(s.get("children", []), level + 1)

    emit(sections, 0)
    doc.build(story)
