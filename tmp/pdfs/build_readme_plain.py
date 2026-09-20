from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "README.md"
OUTPUT = ROOT / "output" / "pdf" / "MedBoard_AI_README.pdf"

BLACK = colors.HexColor("#111111")
DARK_GRAY = colors.HexColor("#444444")
MID_GRAY = colors.HexColor("#777777")
LINE = colors.HexColor("#CCCCCC")
LIGHT_GRAY = colors.HexColor("#F3F3F3")


def clean(text: str) -> str:
    return (
        text.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2192", "->")
        .replace("\u00d7", "x")
    )


def inline(text: str) -> str:
    text = html.escape(clean(text), quote=False)

    def replace_link(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        if target.startswith(("http://", "https://")):
            return f'<link href="{target}" color="#333333">{label}</link>'
        return f'<font color="#333333">{label}</font>'

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace_link, text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" color="#333333">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PlainTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            textColor=BLACK,
            alignment=0,
            spaceAfter=9,
        ),
        "h2": ParagraphStyle(
            "PlainH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=BLACK,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "PlainH3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=BLACK,
            spaceBefore=9,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "PlainBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13.2,
            textColor=BLACK,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "PlainBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.9,
            leading=13,
            textColor=BLACK,
            leftIndent=14,
            firstLineIndent=-8,
            bulletIndent=3,
            spaceAfter=3,
        ),
        "quote": ParagraphStyle(
            "PlainQuote",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13.2,
            textColor=BLACK,
        ),
        "caption": ParagraphStyle(
            "PlainCaption",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=9,
            textColor=DARK_GRAY,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=7,
        ),
        "table": ParagraphStyle(
            "PlainTableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9.1,
            textColor=BLACK,
        ),
        "table_header": ParagraphStyle(
            "PlainTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.3,
            leading=9.2,
            textColor=BLACK,
        ),
        "code": ParagraphStyle(
            "PlainCode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=6.3,
            leading=8.1,
            textColor=BLACK,
        ),
    }


def simple_box(content, *, left_rule: bool = False) -> Table:
    table = Table([[content]], colWidths=[176 * mm], hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]
    if left_rule:
        commands.append(("LINEBEFORE", (0, 0), (0, -1), 2.5, DARK_GRAY))
    table.setStyle(TableStyle(commands))
    return table


def code_box(code: str, style: ParagraphStyle) -> Table:
    code = clean(code.rstrip())
    max_len = max((len(line) for line in code.splitlines()), default=0)
    font_size = 5.7 if max_len > 85 else 6.3
    code_style = ParagraphStyle(
        "DynamicPlainCode",
        parent=style,
        fontSize=font_size,
        leading=font_size + 1.8,
    )
    return simple_box(Preformatted(code, code_style))


def md_table(rows: list[list[str]], s: dict[str, ParagraphStyle]) -> Table:
    columns = len(rows[0])
    if columns == 2:
        widths = [48 * mm, 128 * mm]
    elif columns == 3:
        widths = [45 * mm, 105 * mm, 26 * mm]
    else:
        widths = [176 * mm / columns] * columns
    data = []
    for row_index, row in enumerate(rows):
        cell_style = s["table_header"] if row_index == 0 else s["table"]
        data.append([Paragraph(inline(cell.strip()), cell_style) for cell in row])
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E2E2")),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in range(2, len(data), 2):
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F8F8F8")))
    table.setStyle(TableStyle(commands))
    return table


def parse_markdown(s: dict[str, ParagraphStyle]) -> list[object]:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story: list[object] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            story.append(Paragraph(inline(" ".join(line.strip() for line in paragraph)), s["body"]))
            paragraph.clear()

    i = 0
    while i < len(lines):
        raw = clean(lines[i])
        stripped = raw.strip()

        if not stripped:
            flush()
            i += 1
            continue

        if stripped.startswith("```"):
            flush()
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            story.extend([code_box("\n".join(code), s["code"]), Spacer(1, 6)])
            continue

        if stripped.startswith("!["):
            flush()
            match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
            if match:
                alt, target = match.groups()
                path = ROOT / target
                if path.exists():
                    image = Image(str(path))
                    image._restrictSize(176 * mm, 104 * mm)
                    image.hAlign = "CENTER"
                    story.extend([image, Paragraph(inline(alt), s["caption"])])
            i += 1
            continue

        if stripped.startswith("# "):
            flush()
            story.append(Paragraph(inline(stripped[2:]), s["title"]))
            i += 1
            continue

        if stripped.startswith("## "):
            flush()
            story.append(Paragraph(inline(stripped[3:]), s["h2"]))
            i += 1
            continue

        if stripped.startswith("### "):
            flush()
            story.append(Paragraph(inline(stripped[4:]), s["h3"]))
            i += 1
            continue

        if stripped.startswith(">"):
            flush()
            quote: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(clean(lines[i].strip()[1:].strip()))
                i += 1
            story.extend(
                [
                    simple_box(Paragraph(inline(" ".join(quote)), s["quote"]), left_rule=True),
                    Spacer(1, 6),
                ]
            )
            continue

        if stripped.startswith("|") and i + 1 < len(lines):
            separator = lines[i + 1].strip()
            separator_text = separator.replace("|", "").replace(":", "").replace("-", "").replace(" ", "")
            if separator.startswith("|") and not separator_text:
                flush()
                rows: list[list[str]] = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                    is_separator = all(
                        not cell.replace(":", "").replace("-", "").replace(" ", "")
                        for cell in cells
                    )
                    if not is_separator:
                        rows.append(cells)
                    i += 1
                story.extend([md_table(rows, s), Spacer(1, 7)])
                continue

        if stripped.startswith("- "):
            flush()
            item = stripped[2:]
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() and not lines[i].strip().startswith("-"):
                item += " " + clean(lines[i].strip())
                i += 1
            story.append(Paragraph(inline(item), s["bullet"], bulletText="-"))
            continue

        paragraph.append(raw)
        i += 1

    flush()
    return story


def decorate(canvas, doc) -> None:
    width, height = A4
    canvas.saveState()
    canvas.setTitle("MedBoard AI README")
    canvas.setAuthor("Sabbir Hossain")
    canvas.setSubject("PDF conversion of the repository README")
    canvas.setFillColor(DARK_GRAY)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(17 * mm, height - 12 * mm, "MedBoard AI - README")
    canvas.drawRightString(width - 17 * mm, height - 12 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.line(17 * mm, height - 16 * mm, width - 17 * mm, height - 16 * mm)
    canvas.line(17 * mm, 14 * mm, width - 17 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(17 * mm, 8.5 * mm, "Source: README.md")
    canvas.restoreState()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=24 * mm,
        bottomMargin=20 * mm,
        title="MedBoard AI README",
        author="Sabbir Hossain",
        subject="PDF conversion of the repository README",
        pageCompression=1,
    )
    document.build(parse_markdown(styles()), onFirstPage=decorate, onLaterPages=decorate)
    print(OUTPUT)


if __name__ == "__main__":
    main()
