from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import base64
import os
import tempfile

try:
    from valhalla_strategy_engine import generate_dynamic_tax_strategy
except Exception:
    generate_dynamic_tax_strategy = None

try:
    from valhalla_roi_engine import generate_roi_analysis
except Exception:
    generate_roi_analysis = None

BRAND_RED = "981E26"
BRAND_DARK = "242424"
BRAND_GOLD = "B7892B"
INK = "1F2933"
TEXT_GRAY = "5C6670"
LIGHT_GRAY = "F3F5F7"
LINE_GRAY = "D8DEE5"
LIGHT_RED = "F8ECEE"
LIGHT_GOLD = "FFF8E6"
SOFT_BLUE = "EDF3F8"
WHITE = "FFFFFF"
LOGO_ASSET = os.path.join(os.path.dirname(__file__), "assets", "valhalla_gold_logo_report.b64")


def _money(value):
    try:
        amount = float(value)
        if amount < 0:
            return f"(${abs(amount):,.0f})"
        return f"${amount:,.0f}"
    except Exception:
        return "$0"


def _num(value, default=0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _pct(numerator, denominator):
    denominator = _num(denominator)
    if not denominator:
        return "0.0%"
    return f"{(_num(numerator) / denominator * 100):.1f}%"


def _filing_key(value):
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "single": "single",
        "s": "single",
        "mfj": "married_filing_jointly",
        "married_filing_jointly": "married_filing_jointly",
        "married_joint": "married_filing_jointly",
        "married": "married_filing_jointly",
        "mfs": "married_filing_separately",
        "married_filing_separately": "married_filing_separately",
        "hoh": "head_of_household",
        "head_of_household": "head_of_household",
    }
    return aliases.get(raw, raw or "single")


def _ordinary_brackets(filing_status):
    brackets = {
        "single": [
            (0, 11925, 0.10),
            (11925, 48475, 0.12),
            (48475, 103350, 0.22),
            (103350, 197300, 0.24),
            (197300, 250525, 0.32),
            (250525, 626350, 0.35),
            (626350, float("inf"), 0.37),
        ],
        "married_filing_jointly": [
            (0, 23850, 0.10),
            (23850, 96950, 0.12),
            (96950, 206700, 0.22),
            (206700, 394600, 0.24),
            (394600, 501050, 0.32),
            (501050, 751600, 0.35),
            (751600, float("inf"), 0.37),
        ],
        "head_of_household": [
            (0, 17000, 0.10),
            (17000, 64850, 0.12),
            (64850, 103350, 0.22),
            (103350, 197300, 0.24),
            (197300, 250500, 0.32),
            (250500, 626350, 0.35),
            (626350, float("inf"), 0.37),
        ],
        "married_filing_separately": [
            (0, 11925, 0.10),
            (11925, 48475, 0.12),
            (48475, 103350, 0.22),
            (103350, 197300, 0.24),
            (197300, 250525, 0.32),
            (250525, 375800, 0.35),
            (375800, float("inf"), 0.37),
        ],
    }
    return brackets.get(_filing_key(filing_status), brackets["single"])


def _marginal_rate(taxable_income, filing_status):
    taxable = max(0, _num(taxable_income))
    for lower, upper, rate in _ordinary_brackets(filing_status):
        if lower <= taxable < upper:
            return rate
    return 0.37


def _tax_on_ordinary_income(taxable_income, filing_status):
    taxable = max(0, _num(taxable_income))
    tax = 0
    for lower, upper, rate in _ordinary_brackets(filing_status):
        if taxable <= lower:
            break
        amount = min(taxable, upper) - lower
        tax += max(0, amount) * rate
    return tax


def _ltcg_thresholds(filing_status):
    thresholds = {
        "single": (48350, 533400),
        "married_filing_jointly": (96700, 600050),
        "head_of_household": (64750, 566700),
        "married_filing_separately": (48350, 300000),
    }
    return thresholds.get(_filing_key(filing_status), thresholds["single"])


def _safe(value, fallback="Not provided"):
    if value is None or value == "":
        return fallback
    return str(value)


def _limit(text, length=420):
    text = _safe(text, "")
    if len(text) <= length:
        return text
    return text[: length - 3].rstrip() + "..."


def _set_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def _set_border(cell, color=LINE_GRAY, size="4"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right"):
        element = borders.find(qn("w:" + edge))
        if element is None:
            element = OxmlElement("w:" + edge)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def _cell_margins(cell, top=120, start=140, bottom=120, end=140):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn("w:" + name))
        if node is None:
            node = OxmlElement("w:" + name)
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def _format_cell(cell, fill=None, color=INK, bold=False, size=8.8, align=None, border=LINE_GRAY):
    if fill:
        _set_shading(cell, fill)
    _set_border(cell, border)
    _cell_margins(cell)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for paragraph in cell.paragraphs:
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.05
        if align is not None:
            paragraph.alignment = align
        for run in paragraph.runs:
            run.font.name = "Aptos"
            run.font.size = Pt(size)
            run.bold = bold
            run.font.color.rgb = RGBColor.from_string(color)


def _paragraph(paragraph, size=9.4, color=INK, bold=False, italic=False, before=0, after=5, align=None):
    paragraph.paragraph_format.space_before = Pt(before)
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.08
    if align is not None:
        paragraph.alignment = align
    for run in paragraph.runs:
        run.font.name = "Aptos"
        run.font.size = Pt(size)
        run.font.color.rgb = RGBColor.from_string(color)
        if bold:
            run.bold = True
        if italic:
            run.italic = True


def _set_widths(table, widths):
    table.autofit = False
    for row in table.rows:
        for idx, width in enumerate(widths):
            if idx < len(row.cells):
                row.cells[idx].width = Inches(width)


def _page_number(paragraph):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)


def _logo_temp_path():
    if not os.path.exists(LOGO_ASSET):
        return None
    try:
        with open(LOGO_ASSET, "r", encoding="utf-8") as logo_file:
            encoded = logo_file.read().strip()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(base64.b64decode(encoded))
            return tmp.name
    except Exception:
        return None


def _add_logo(paragraph, width=3.4):
    logo_path = _logo_temp_path()
    if not logo_path:
        return False
    try:
        paragraph.add_run().add_picture(logo_path, width=Inches(width))
        return True
    except Exception:
        return False
    finally:
        try:
            os.remove(logo_path)
        except Exception:
            pass


def _configure(doc, data):
    section = doc.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.58)
    section.bottom_margin = Inches(0.62)
    section.left_margin = Inches(0.62)
    section.right_margin = Inches(0.62)

    header = section.header.paragraphs[0]
    header.text = f"Valhalla Premium Tax Strategy Report | {_safe(data.get('client_name'), 'Client')} | {_safe(data.get('tax_year'), 'Tax Year')}"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _paragraph(header, size=7.5, color=TEXT_GRAY, after=0)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Valhalla Tax & Finance LLC | Confidential planning document | Page ")
    _page_number(footer)
    _paragraph(footer, size=7.3, color=TEXT_GRAY, after=0)

    doc.styles["Normal"].font.name = "Aptos"
    doc.styles["Normal"].font.size = Pt(9.4)


def _section_title(doc, title, kicker=None):
    if kicker:
        k = doc.add_paragraph(kicker.upper())
        _paragraph(k, size=7.3, color=BRAND_GOLD, bold=True, before=5, after=0)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(title)
    r.font.name = "Aptos Display"
    r.font.size = Pt(15.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor.from_string(BRAND_RED)
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "7")
    bottom.set(qn("w:space"), "2")
    bottom.set(qn("w:color"), LINE_GRAY)
    border.append(bottom)
    p._p.get_or_add_pPr().append(border)


def _label_value(cell, label, value, value_color=INK, value_size=14.0):
    p = cell.paragraphs[0]
    r = p.add_run(label.upper() + "\n")
    r.font.name = "Aptos"
    r.font.size = Pt(7.1)
    r.font.bold = True
    r.font.color.rgb = RGBColor.from_string(TEXT_GRAY)
    v = p.add_run(str(value))
    v.font.name = "Aptos Display"
    v.font.size = Pt(value_size)
    v.font.bold = True
    v.font.color.rgb = RGBColor.from_string(value_color)


def _callout(doc, title, body, fill=LIGHT_RED, accent=BRAND_RED):
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_widths(table, [0.14, 7.12])
    bar = table.cell(0, 0)
    body_cell = table.cell(0, 1)
    _set_shading(bar, accent)
    _set_border(bar, accent, "0")
    _cell_margins(bar, 40, 20, 40, 20)
    _format_cell(body_cell, fill=fill, border="E5D5D8")
    p = body_cell.paragraphs[0]
    p.add_run(title)
    _paragraph(p, size=9.6, color=accent, bold=True, after=2)
    p2 = body_cell.add_paragraph(_limit(body, 560))
    _paragraph(p2, size=8.9, color=INK, after=0)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def _simple_table(doc, rows, widths, header_fill=BRAND_RED, font_size=8.3):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_widths(table, widths)
    for r_idx, row in enumerate(rows):
        for c_idx, value in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = str(value)
            header = r_idx == 0
            fill = header_fill if header else (WHITE if r_idx % 2 else LIGHT_GRAY)
            color = WHITE if header else INK
            bold = header or c_idx == 0
            align = WD_ALIGN_PARAGRAPH.CENTER if header or c_idx in (0, 1) else WD_ALIGN_PARAGRAPH.LEFT
            _format_cell(cell, fill=fill, color=color, bold=bold, size=font_size, align=align)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return table


def _refund_or_due(data):
    balance = _num(data.get("balance_due", 0))
    refund = _num(data.get("refund", 0))
    if balance > 0:
        return "Balance due", _money(-balance), BRAND_RED
    if refund > 0:
        return "Refund", _money(refund), BRAND_RED
    return "Refund / balance", "$0", BRAND_DARK


def _opportunity_range(data, actions):
    total = 0
    for action in actions or []:
        savings = action.get("estimated_savings") or action.get("tax_impact") or action.get("impact")
        if isinstance(savings, (int, float)):
            total += float(savings)
    if total:
        return _money(total), _money(total * 1.6)
    taxable = _num(data.get("taxable_income", 0))
    sched_c = _num(data.get("schedule_c_net_profit", 0))
    if sched_c > 0:
        return "$2,500", "$8,000+"
    if taxable > 0:
        return "$1,000", "$4,000+"
    return "$500", "$2,500+"


def _tax_efficiency_score(data, actions):
    explicit = data.get("tax_efficiency_score") or data.get("tax_efficiency")
    if isinstance(explicit, dict):
        explicit = explicit.get("score") or explicit.get("overall_score")
    if explicit not in (None, ""):
        try:
            return max(1, min(100, int(float(explicit))))
        except Exception:
            pass

    score = 72
    taxable = _num(data.get("taxable_income", 0))
    total_tax = _num(data.get("total_tax", 0))
    agi = _num(data.get("agi", 0))
    refund = _num(data.get("refund", 0))
    balance = _num(data.get("balance_due", 0))
    sched_c = _num(data.get("schedule_c_net_profit", 0))

    if agi and total_tax / agi < 0.08:
        score += 6
    if taxable and taxable < 100000:
        score += 4
    if balance > 2500:
        score -= 8
    if refund > 4000:
        score -= 5
    if sched_c > 0:
        score -= 4
    if actions:
        score += min(6, len(actions))
    return max(40, min(95, int(score)))


def _score_band(score):
    if score >= 85:
        return "Strong", BRAND_GOLD, "Solid tax position; focus on preserving gains and executing the highest-value actions."
    if score >= 70:
        return "Good with opportunities", BRAND_RED, "Planning value exists; prioritize the actions that improve cash flow, deductions, and long-term flexibility."
    return "Needs attention", BRAND_RED, "Several controllable planning items should be addressed before year-end."


def _default_actions(data):
    actions = []
    balance = _num(data.get("balance_due", 0))
    refund = _num(data.get("refund", 0))
    sched_c = _num(data.get("schedule_c_net_profit", 0))
    taxable = _num(data.get("taxable_income", 0))
    state = str(data.get("state", "")).upper()

    if sched_c > 0:
        actions.append({
            "title": "Fund a self-employed retirement plan",
            "estimated_savings": "Potential annual federal savings from deductible contributions",
            "timeline": "Before plan and contribution deadlines",
            "reason": "Schedule C profit creates one of the cleanest recurring planning opportunities: deductible retirement funding that also builds long-term assets.",
        })
        actions.append({
            "title": "Review Schedule C documentation and entity timing",
            "estimated_savings": "Risk reduction and future payroll-tax planning",
            "timeline": "Next 30 to 90 days",
            "reason": "Business profit should be supported by clean records, substantiated expenses, and an annual review of whether entity changes are warranted.",
        })

    if balance > 0:
        actions.append({
            "title": "Correct withholding or estimated tax payments",
            "estimated_savings": "Cash-flow improvement and penalty prevention",
            "timeline": "Immediately",
            "reason": f"The return reflects a balance due of {_money(balance)}. Adjusting payroll withholding or estimates can prevent repeating the shortfall.",
        })
    elif refund > 2500:
        actions.append({
            "title": "Right-size withholding for better cash flow",
            "estimated_savings": "Cash-flow improvement",
            "timeline": "Next payroll cycle",
            "reason": f"The return reflects a refund of {_money(refund)}. Some excess withholding may be redirected toward planning goals during the year.",
        })

    if taxable > 0:
        actions.append({
            "title": "Model bracket-aware Roth and investment planning",
            "estimated_savings": "Long-term tax flexibility",
            "timeline": "Annual review",
            "reason": "Taxable income creates the need to coordinate Roth conversions, capital gains, retirement contributions, and future income timing.",
        })

    if state == "AZ":
        actions.append({
            "title": "Use Arizona credits and deduction timing intentionally",
            "estimated_savings": "State tax reduction opportunity",
            "timeline": "Before year-end",
            "reason": "Arizona planning can add value through charitable credits, school credits, and careful year-end payment timing.",
        })

    if not actions:
        actions.append({
            "title": "Build a year-round tax planning calendar",
            "estimated_savings": "Planning discipline and risk reduction",
            "timeline": "Next 30 days",
            "reason": "The return should become the starting point for proactive planning, not merely a filing record.",
        })
    return actions[:5]


def _normalize_actions(data, strategy_output, roi_output):
    source = data.get("priority_actions") or strategy_output.get("priority_actions") or []
    actions = []
    for item in source:
        if isinstance(item, str):
            actions.append({"title": item, "estimated_savings": "Planning value", "timeline": "Review", "reason": item})
        elif isinstance(item, dict):
            actions.append({
                "title": item.get("title") or item.get("strategy") or item.get("recommendation") or "Planning action",
                "estimated_savings": item.get("estimated_savings") or item.get("tax_impact") or item.get("impact") or "Planning value",
                "timeline": item.get("timeline") or item.get("timing") or "Review",
                "reason": item.get("reason") or item.get("advisor_reasoning") or item.get("why") or "Client-specific planning opportunity.",
                "score": item.get("score") or item.get("priority_score"),
            })

    for item in data.get("roi_strategies") or roi_output.get("roi_strategies") or []:
        if isinstance(item, dict):
            actions.append({
                "title": item.get("strategy") or item.get("title") or "ROI strategy",
                "estimated_savings": _money(item.get("estimated_savings", 0)) if item.get("estimated_savings") is not None else "Quantified opportunity",
                "timeline": item.get("timeline") or "Review",
                "reason": item.get("advisor_reasoning") or "Quantified planning opportunity identified.",
                "score": item.get("score"),
            })

    if not actions:
        actions = _default_actions(data)
    actions.sort(key=lambda x: _num(x.get("score", 0)), reverse=True)
    return actions[:6]


def _augment_report_actions(data, actions):
    titles = " | ".join(_safe(action.get("title"), "").lower() for action in actions)
    augmented = list(actions)
    if "roth contribution" not in titles and "roth conversion" not in titles:
        annual = _roth_projection(data)[0]["annual"]
        augmented.append({
            "title": "Roth contribution / conversion modeling",
            "estimated_savings": f"1/3/5-year Roth reserve modeled from {_money(annual)} annual capacity",
            "timeline": "Annual bracket review",
            "reason": "Model Roth funding, conversion tax cost, and future tax-free reserve before choosing between current deductions and long-term tax diversification.",
            "score": 88,
        })
    if "capital" not in titles and "gain" not in titles and "loss" not in titles:
        plan = _capital_gain_plan(data)
        augmented.append({
            "title": "Capital gain and loss harvesting review",
            "estimated_savings": f"{_money(plan['zero_room'])} estimated 0% LTCG room; {_money(plan['loss_tax_value'])} loss-harvest value",
            "timeline": "Before taxable trades and year-end",
            "reason": "Review unrealized gains, losses, fund distributions, and charitable gifting candidates so portfolio decisions are coordinated with tax brackets.",
            "score": 84,
        })
    augmented.sort(key=lambda x: _num(x.get("score", 0)), reverse=True)
    return augmented[:7]


def _cover(doc, data, actions):
    logo = doc.add_paragraph()
    logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if not _add_logo(logo, width=3.55):
        logo.add_run("VALHALLA TAX & FINANCE LLC")
        _paragraph(logo, size=15, color=BRAND_RED, bold=True, after=3)
    logo.paragraph_format.space_after = Pt(12)

    band = doc.add_table(rows=1, cols=1)
    band.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_widths(band, [7.25])
    cell = band.cell(0, 0)
    _format_cell(cell, fill=BRAND_DARK, color=WHITE, border=BRAND_DARK)
    p = cell.paragraphs[0]
    p.add_run("Premium tax planning report | Client advisory deliverable")
    _paragraph(p, size=9.2, color="D7DEE6", bold=True, after=0, align=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    title = doc.add_paragraph()
    title.add_run("Tax Strategy\nImplementation Report")
    _paragraph(title, size=28, color=BRAND_RED, bold=True, after=4)

    subtitle = doc.add_paragraph("A premium client-facing roadmap for reducing avoidable tax drag, improving cash-flow control, and turning the tax return into an implementation plan.")
    _paragraph(subtitle, size=10.8, color=TEXT_GRAY, after=13)

    meta = doc.add_table(rows=1, cols=4)
    meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_widths(meta, [2.15, 1.4, 1.55, 2.15])
    score = _tax_efficiency_score(data, actions)
    band_label, band_color, _ = _score_band(score)
    values = [
        ("Client", _safe(data.get("client_name"), "Client"), BRAND_RED),
        ("Tax Year", _safe(data.get("tax_year"), "2025"), INK),
        ("Efficiency", f"{score}/100", band_color),
        ("Prepared By", "Scott Kunz, ChFC, TPCP, EA", INK),
    ]
    for idx, (label, value, color) in enumerate(values):
        meta_cell = meta.cell(0, idx)
        _format_cell(meta_cell, fill=LIGHT_GRAY, border=LINE_GRAY)
        _label_value(meta_cell, label, value, color, 11.6)

    low, high = _opportunity_range(data, actions)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    _callout(
        doc,
        "Executive Value Thesis",
        f"The strategies in this report point to an estimated annual planning opportunity of approximately {low} to {high}, depending on contribution levels, timing, documentation, and final implementation choices.",
        fill=LIGHT_GOLD,
        accent=BRAND_GOLD,
    )

    summary = data.get("advisor_summary") or data.get("planning_summary") or data.get("top_planning_focus") or "This report turns the reviewed return into an implementation plan: what matters most, why it matters, and what the client should act on first."
    _callout(doc, "Advisor Summary", summary, fill=LIGHT_RED, accent=BRAND_RED)

    toc = doc.add_table(rows=5, cols=2)
    toc.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_widths(toc, [0.52, 6.72])
    items = [
        ("01", "Executive opportunity dashboard"),
        ("02", "Confirmed tax position and planning read"),
        ("03", "Visual planning analysis"),
        ("04", "Priority strategies and action plan"),
        ("05", "Implementation roadmap and advisor notes"),
    ]
    for row, (num, label) in zip(toc.rows, items):
        row.cells[0].text = num
        row.cells[1].text = label
        _format_cell(row.cells[0], fill=BRAND_RED, color=WHITE, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, border=BRAND_RED)
        _format_cell(row.cells[1], fill=WHITE, color=INK, size=9.1)
    doc.add_page_break()


def _dashboard(doc, data, actions):
    _section_title(doc, "Executive Opportunity Dashboard", "Client snapshot")
    low, high = _opportunity_range(data, actions)
    refund_label, refund_value, refund_color = _refund_or_due(data)
    cards = [
        ("AGI", _money(data.get("agi", 0)), "Phaseout and planning baseline", BRAND_RED),
        ("Taxable Income", _money(data.get("taxable_income", 0)), "Bracket management base", INK),
        ("Total Tax", _money(data.get("total_tax", 0)), "Current federal tax cost", INK),
        (refund_label, refund_value, "Cash-flow planning target", refund_color),
        ("Planning Range", f"{low}-{high}", "Estimated annual opportunity", BRAND_GOLD),
    ]
    table = doc.add_table(rows=1, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_widths(table, [1.42, 1.42, 1.42, 1.42, 1.62])
    for idx, (label, value, note, color) in enumerate(cards):
        cell = table.cell(0, idx)
        _format_cell(cell, fill=LIGHT_GRAY, border="E1E5EA")
        _label_value(cell, label, value, color, 11.4 if idx < 4 else 10.5)
        p = cell.add_paragraph(note)
        _paragraph(p, size=7.2, color=TEXT_GRAY, after=0)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    _tax_efficiency_panel(doc, data, actions)
    _strategy_cards(doc, actions[:3], compact=True)


def _tax_efficiency_panel(doc, data, actions):
    score = _tax_efficiency_score(data, actions)
    label, color, read = _score_band(score)
    low, high = _opportunity_range(data, actions)
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_widths(table, [1.3, 5.95])
    score_cell = table.cell(0, 0)
    read_cell = table.cell(0, 1)
    _format_cell(score_cell, fill=color, color=WHITE, bold=True, border=color)
    score_cell.paragraphs[0].text = ""
    p = score_cell.paragraphs[0]
    r = p.add_run(f"{score}\n")
    r.font.name = "Aptos Display"
    r.font.size = Pt(24)
    r.bold = True
    r.font.color.rgb = RGBColor.from_string(WHITE)
    label_run = p.add_run("TAX EFFICIENCY")
    label_run.font.name = "Aptos"
    label_run.font.size = Pt(7.4)
    label_run.bold = True
    label_run.font.color.rgb = RGBColor.from_string(WHITE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)

    _format_cell(read_cell, fill=WHITE, border="DDE3EA")
    h = read_cell.paragraphs[0]
    h.add_run(f"{label} | Priority planning range: {low}-{high}")
    _paragraph(h, size=10.0, color=INK, bold=True, after=2)
    body = read_cell.add_paragraph(read)
    _paragraph(body, size=8.4, color=TEXT_GRAY, after=0)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def _tax_position(doc, data):
    _section_title(doc, "Confirmed Tax Position", "Return facts")
    rows = [
        ["Tax Fact", "Confirmed Amount", "Planning Read"],
        ["Filing status", _safe(data.get("filing_status"), "Not provided"), "Controls brackets, standard deduction, credits, and phaseouts"],
        ["Dependents", _safe(data.get("dependents"), "0"), "Important for credits, family planning, education, and payroll strategies"],
        ["AGI", _money(data.get("agi", 0)), "Baseline for phaseouts, credits, IRMAA, and state planning"],
        ["Taxable income", _money(data.get("taxable_income", 0)), "Primary driver for bracket-aware retirement and investment strategy"],
        ["Total tax", _money(data.get("total_tax", 0)), "Current tax-cost benchmark"],
        [_refund_or_due(data)[0], _refund_or_due(data)[1], "Withholding and estimated payments should be tuned before year-end"],
    ]
    _simple_table(doc, rows, [1.52, 1.78, 3.95], font_size=8.4)

    focus = data.get("top_planning_focus") or "The return should be used as a planning baseline for retirement funding, withholding, deductions, and state-specific opportunities."
    _callout(doc, "Advisor Read", focus, fill=SOFT_BLUE, accent="4D6F8C")


def _business_section(doc, data):
    gross = _num(data.get("schedule_c_gross_revenue", 0))
    net = _num(data.get("schedule_c_net_profit", 0))
    if not gross and not net:
        return
    expenses = max(gross - net, 0)
    _section_title(doc, "Business Owner Planning", "Schedule C")
    rows = [
        ["Metric", "Amount", "Planning Interpretation"],
        ["Gross revenue", _money(gross), "Business activity is large enough to justify proactive tax structure and documentation review"],
        ["Expenses", _money(expenses), f"Expense ratio is {_pct(expenses, gross)}; confirm substantiation and business purpose"],
        ["Net profit", _money(net), f"Net margin is {_pct(net, gross)}; retirement funding and entity review should be modeled"],
        ["Contract labor", _money(data.get("contract_labor", 0)), "Confirm worker classification, W-9 files, and 1099 documentation"],
    ]
    _simple_table(doc, rows, [1.55, 1.45, 4.25], font_size=8.3)
    _callout(doc, "Business Planning Priority", "Tie Schedule C profit to clean books, documented expenses, retirement funding, quarterly tax planning, and an annual entity trigger review.", fill=LIGHT_GOLD, accent=BRAND_GOLD)


def _roth_projection(data):
    filing = _filing_key(data.get("filing_status"))
    taxable = _num(data.get("taxable_income", 0))
    age = _num(data.get("age", 0))
    explicit = data.get("roth_contribution") or data.get("annual_roth_contribution") or data.get("roth_conversion_amount")
    if explicit:
        annual = max(0, _num(explicit))
    else:
        per_taxpayer = 8000 if age >= 50 else 7000
        annual = per_taxpayer * (2 if filing == "married_filing_jointly" else 1)
    annual = min(max(annual, 3500), 50000)
    rate = _marginal_rate(taxable, filing)
    years = [1, 3, 5]
    growth = 0.06
    projections = []
    for year_count in years:
        future_value = 0
        for i in range(year_count):
            future_value += annual * ((1 + growth) ** (year_count - i - 1))
        conversion_tax = annual * rate * year_count
        projected_tax = _tax_on_ordinary_income(taxable + annual, filing) - _tax_on_ordinary_income(taxable, filing)
        projections.append({
            "years": year_count,
            "annual": annual,
            "estimated_tax_cost": max(0, conversion_tax),
            "first_year_tax": max(0, projected_tax),
            "roth_reserve": future_value,
            "tax_free_growth": max(0, future_value - (annual * year_count)),
        })
    return projections


def _capital_gain_plan(data):
    filing = _filing_key(data.get("filing_status"))
    taxable = _num(data.get("taxable_income", data.get("agi", 0)))
    gain = _num(data.get("capital_gains", data.get("capital_gain", data.get("net_capital_gain", 0))))
    loss = _num(data.get("capital_losses", data.get("capital_loss", data.get("net_capital_loss", 0))))
    if gain < 0 and not loss:
        loss = gain
        gain = 0
    zero_threshold, fifteen_threshold = _ltcg_thresholds(filing)
    zero_room = max(0, zero_threshold - taxable)
    fifteen_room = max(0, fifteen_threshold - max(taxable, zero_threshold))
    rate = _marginal_rate(taxable, filing)
    deductible_loss = min(abs(loss), 3000) if loss < 0 else 3000
    loss_tax_value = deductible_loss * rate
    harvest_gain_value = min(max(gain, zero_room), zero_room) * 0.15 if gain > 0 else zero_room * 0.15
    return {
        "gain": gain,
        "loss": loss,
        "zero_room": zero_room,
        "fifteen_room": fifteen_room,
        "loss_tax_value": loss_tax_value,
        "harvest_gain_value": harvest_gain_value,
        "marginal_rate": rate,
    }


def _projection_chart(doc, title, periods, tax_costs, values, caption, width=5.95):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        chart_path = tmp.name
    try:
        if not _create_projection_image(chart_path, title, periods, tax_costs, values):
            return
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(4)
        p.add_run().add_picture(chart_path, width=Inches(width))
        cap = doc.add_paragraph(caption)
        _paragraph(cap, size=7.8, color=TEXT_GRAY, italic=True, after=7, align=WD_ALIGN_PARAGRAPH.CENTER)
    except Exception:
        return
    finally:
        try:
            os.remove(chart_path)
        except Exception:
            pass


def _roth_strategy_section(doc, data):
    projections = _roth_projection(data)
    filing = _safe(data.get("filing_status"), "Not provided")
    rate = _marginal_rate(data.get("taxable_income", 0), data.get("filing_status", "single"))
    annual = projections[0]["annual"] if projections else 0
    _section_title(doc, "Roth Contribution and Conversion Strategy", "Retirement tax leverage")
    intro = doc.add_paragraph(
        "This section shows the client why retirement tax planning matters now: the decision is not only the current-year deduction, but the future tax-free reserve that can be built when Roth funding or bracket-aware conversions are modeled intentionally."
    )
    _paragraph(intro, size=8.8, color=INK, after=5)
    rows = [
        ["Planning Lever", "Illustrated Amount", "Client Value"],
        ["Annual Roth funding / conversion target", _money(annual), "Creates a visible funding target instead of a vague retirement recommendation"],
        ["Current marginal bracket estimate", f"{rate:.0%}", f"Used only to illustrate the tax cost of filling Roth capacity for a {filing} taxpayer"],
        ["1-year projected Roth reserve", _money(projections[0]["roth_reserve"]), "Shows immediate funded value"],
        ["5-year projected Roth reserve", _money(projections[-1]["roth_reserve"]), "Frames long-term tax-free accumulation potential"],
    ]
    _simple_table(doc, rows, [2.05, 1.65, 3.55], font_size=8.15)
    _projection_chart(
        doc,
        "Roth 1 / 3 / 5-Year Illustration",
        [f"{p['years']} yr" for p in projections],
        [p["estimated_tax_cost"] for p in projections],
        [p["roth_reserve"] for p in projections],
        "Illustrative model: annual Roth funding/conversion target compared with estimated ordinary-income tax cost and projected Roth reserve at a 6% growth assumption.",
        width=5.85,
    )
    _callout(
        doc,
        "Advisor Talking Point",
        "The Roth strategy should be reviewed alongside cash flow, contribution eligibility, conversion capacity, state tax impact, and retirement time horizon before implementation.",
        fill=SOFT_BLUE,
        accent="4D6F8C",
    )


def _capital_gain_strategy_section(doc, data):
    plan = _capital_gain_plan(data)
    _section_title(doc, "Capital Gain and Loss Strategy", "Investment tax planning")
    rows = [
        ["Planning Lever", "Amount", "Recommended Review"],
        ["Estimated 0% LTCG bracket room", _money(plan["zero_room"]), "Harvest qualified long-term gains intentionally if the client has low-rate capacity"],
        ["15% LTCG bracket capacity", _money(plan["fifteen_room"]), "Coordinate asset sales, income timing, and charitable strategies before year-end"],
        ["Loss-harvesting tax value", _money(plan["loss_tax_value"]), "Use realized losses to offset gains or up to $3,000 of ordinary income when available"],
        ["Current capital gain/loss shown", _money(plan["gain"] or plan["loss"]), "Verify brokerage statements before acting"],
    ]
    _simple_table(doc, rows, [2.1, 1.45, 3.7], font_size=8.15)
    _chart(
        doc,
        "Capital Gain / Loss Planning Capacity",
        ["0% gain room", "15% gain capacity", "Loss tax value"],
        [plan["zero_room"], plan["fifteen_room"], plan["loss_tax_value"]],
        "This chart turns portfolio tax planning into a client-visible opportunity: gain harvesting, loss harvesting, and timing should be reviewed before taxable trades are made.",
        kind="barh",
        width=5.85,
    )
    _callout(
        doc,
        "Capital Gains Action",
        "Before year-end, review unrealized gains and losses, mutual fund distributions, charitable gifting candidates, and income timing so portfolio decisions are made with tax brackets in view.",
        fill=LIGHT_GOLD,
        accent=BRAND_GOLD,
    )


def _pil_font(size, bold=False):
    try:
        from PIL import ImageFont
        names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf", "calibri.ttf"]
        for name in names:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                pass
        return ImageFont.load_default()
    except Exception:
        return None


def _create_pil_chart(path, title, labels, values, kind="barh"):
    try:
        from PIL import Image, ImageDraw

        values = [_num(v) for v in values]
        if not labels or not values:
            return False
        width, height = 1120, 430
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        title_font = _pil_font(28, True)
        label_font = _pil_font(19)
        small_font = _pil_font(17)
        draw.text((32, 22), title, fill="#242424", font=title_font)
        colors = ["#981E26", "#B7892B", "#334E68", "#627D98", "#D8DEE5", "#5C6670"]
        max_value = max(max(abs(v) for v in values), 1)

        if kind == "bar":
            left, bottom, chart_w, chart_h = 70, 342, 990, 230
            bar_w = max(46, int(chart_w / max(len(values) * 2.2, 1)))
            gap = (chart_w - (bar_w * len(values))) / max(len(values) + 1, 1)
            for idx, (label, value) in enumerate(zip(labels, values)):
                x0 = int(left + gap + idx * (bar_w + gap))
                bar_h = int((abs(value) / max_value) * chart_h)
                y0 = bottom - bar_h
                draw.rectangle([x0, y0, x0 + bar_w, bottom], fill=colors[idx % len(colors)])
                draw.text((x0 - 8, bottom + 12), str(label)[:15], fill="#1F2933", font=small_font)
                draw.text((x0 - 8, max(70, y0 - 24)), f"{value:,.0f}", fill="#5C6670", font=small_font)
            draw.line([left, bottom, left + chart_w, bottom], fill="#D8DEE5", width=2)
        else:
            left, top, bar_w, row_h = 290, 86, 735, 54
            for idx, (label, value) in enumerate(zip(labels, values)):
                y = top + idx * row_h
                length = int((abs(value) / max_value) * bar_w)
                draw.text((32, y + 10), str(label)[:24], fill="#1F2933", font=label_font)
                draw.rectangle([left, y + 8, left + length, y + 34], fill=colors[idx % len(colors)])
                draw.text((left + length + 12, y + 8), _money(value), fill="#5C6670", font=small_font)
            draw.line([left, top - 8, left, top + row_h * len(values)], fill="#D8DEE5", width=2)

        image.save(path, "PNG")
        return True
    except Exception:
        return False


def _create_projection_image(path, title, periods, tax_costs, values):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        import numpy as np

        x = np.arange(len(periods))
        fig, ax = plt.subplots(figsize=(6.35, 2.55))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.bar(x - 0.18, tax_costs, 0.36, label="Estimated tax cost", color="#981E26")
        ax.bar(x + 0.18, values, 0.36, label="Projected Roth reserve", color="#B7892B")
        ax.set_title(title, fontsize=10.4, fontweight="bold", color="#242424", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, fontsize=8.2, color="#1F2933")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
        ax.tick_params(axis="y", labelsize=7.5, colors="#5C6670")
        ax.grid(axis="y", color="#E6E9ED", linewidth=0.8)
        ax.legend(loc="upper left", frameon=False, fontsize=7.4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=1.0)
        fig.savefig(path, dpi=175, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        try:
            from PIL import Image, ImageDraw

            width, height = 1120, 455
            image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(image)
            title_font = _pil_font(28, True)
            label_font = _pil_font(19)
            small_font = _pil_font(17)
            draw.text((32, 22), title, fill="#242424", font=title_font)
            max_value = max(max([_num(v) for v in list(tax_costs) + list(values)]), 1)
            left, bottom, chart_w, chart_h = 95, 350, 930, 230
            group_w = chart_w / max(len(periods), 1)
            for idx, period in enumerate(periods):
                x = int(left + idx * group_w + 46)
                tax_h = int((_num(tax_costs[idx]) / max_value) * chart_h)
                value_h = int((_num(values[idx]) / max_value) * chart_h)
                draw.rectangle([x, bottom - tax_h, x + 58, bottom], fill="#981E26")
                draw.rectangle([x + 68, bottom - value_h, x + 126, bottom], fill="#B7892B")
                draw.text((x + 18, bottom + 12), str(period), fill="#1F2933", font=label_font)
                draw.text((x - 8, max(70, bottom - tax_h - 24)), _money(tax_costs[idx]), fill="#981E26", font=small_font)
                draw.text((x + 62, max(70, bottom - value_h - 24)), _money(values[idx]), fill="#B7892B", font=small_font)
            draw.rectangle([36, 385, 58, 407], fill="#981E26")
            draw.text((68, 384), "Estimated tax cost", fill="#5C6670", font=small_font)
            draw.rectangle([275, 385, 297, 407], fill="#B7892B")
            draw.text((307, 384), "Projected Roth reserve", fill="#5C6670", font=small_font)
            draw.line([left, bottom, left + chart_w, bottom], fill="#D8DEE5", width=2)
            image.save(path, "PNG")
            return True
        except Exception:
            return False


def _create_chart(path, title, labels, values, kind="barh"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        values = [_num(v) for v in values]
        colors = ["#981E26", "#B7892B", "#334E68", "#627D98", "#D8DEE5", "#5C6670"]
        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(6.2, 2.25))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        if kind == "donut":
            safe = [max(v, 0) for v in values]
            total = sum(safe) or 1
            ax.pie(safe, labels=labels, startangle=90, colors=colors[: len(labels)], wedgeprops={"width": 0.45, "edgecolor": "white"}, textprops={"fontsize": 7.2})
            ax.text(0, 0, _money(total), ha="center", va="center", fontsize=10, fontweight="bold", color="#981E26")
        elif kind == "bar":
            ax.bar(labels, values, color=colors[: len(labels)], width=0.56)
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
            ax.tick_params(axis="x", labelsize=7.2, rotation=18, colors="#1F2933")
            ax.tick_params(axis="y", labelsize=7.4, colors="#5C6670")
            ax.grid(axis="y", color="#E6E9ED", linewidth=0.8)
        else:
            order = list(range(len(labels)))[::-1]
            ax.barh([labels[i] for i in order], [values[i] for i in order], color=[colors[i % len(colors)] for i in order], height=0.54)
            ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
            ax.tick_params(axis="x", labelsize=7.3, colors="#5C6670")
            ax.tick_params(axis="y", labelsize=8.0, colors="#1F2933")
            ax.grid(axis="x", color="#E6E9ED", linewidth=0.8)

        ax.set_title(title, fontsize=10.2, fontweight="bold", color="#242424", pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=1.0)
        fig.savefig(path, dpi=175, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        return _create_pil_chart(path, title, labels, values, kind)


def _chart(doc, title, labels, values, caption, kind="barh", width=5.95):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        chart_path = tmp.name
    try:
        if _create_chart(chart_path, title, labels, values, kind=kind):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(4)
            p.add_run().add_picture(chart_path, width=Inches(width))
            cap = doc.add_paragraph(caption)
            _paragraph(cap, size=7.8, color=TEXT_GRAY, italic=True, after=7, align=WD_ALIGN_PARAGRAPH.CENTER)
    finally:
        try:
            os.remove(chart_path)
        except Exception:
            pass


def _planning_value_stack(data):
    roth = _roth_projection(data)
    cap = _capital_gain_plan(data)
    sched_c = max(0, _num(data.get("schedule_c_net_profit", 0)))
    rate = _marginal_rate(data.get("taxable_income", 0), data.get("filing_status", "single"))
    refund = _num(data.get("refund", 0))
    balance = _num(data.get("balance_due", 0))
    values = [
        ("Roth reserve", roth[-1]["roth_reserve"] if roth else 0),
        ("Cap gain room", min(cap["zero_room"], 50000)),
        ("Loss harvest value", cap["loss_tax_value"]),
        ("Retirement deduction", min(sched_c * rate, 12000) if sched_c > 0 else 0),
        ("Cash-flow tuning", max(refund, balance)),
    ]
    return [(label, value) for label, value in values if value > 0]


def _visuals(doc, data, actions):
    _section_title(doc, "Planning Visuals", "Data read")
    _chart(
        doc,
        "Current Federal Tax Position",
        ["AGI", "Taxable Income", "Total Tax", "Withholding", _refund_or_due(data)[0]],
        [data.get("agi", 0), data.get("taxable_income", 0), data.get("total_tax", 0), data.get("federal_withholding", 0), abs(_num(data.get("balance_due", 0)) or _num(data.get("refund", 0)))],
        "This chart frames the planning baseline: income, taxable exposure, federal tax cost, payments, and the year-end cash-flow result.",
        kind="barh",
        width=5.75,
    )

    gross = _num(data.get("schedule_c_gross_revenue", 0))
    net = _num(data.get("schedule_c_net_profit", 0))
    if gross or net:
        _chart(
            doc,
            "Schedule C Economics",
            ["Gross Revenue", "Expenses", "Net Profit"],
            [gross, max(gross - net, 0), net],
            "Business-owner planning should focus on clean records, retirement contribution modeling, and quarterly tax discipline.",
            kind="barh",
            width=5.75,
        )

    labels = [a.get("title", "Strategy")[:24] for a in actions[:5]]
    values = []
    for idx, action in enumerate(actions[:5], start=1):
        raw = action.get("score")
        values.append(_num(raw, max(30, 90 - idx * 10)))
    _chart(doc, "Priority Strategy Ranking", labels, values, "Relative ranking based on urgency, tax impact, implementation timing, and planning value.", kind="barh", width=5.75)

    stack = _planning_value_stack(data)
    if stack:
        _chart(
            doc,
            "Planning Value Stack",
            [label for label, _ in stack[:5]],
            [value for _, value in stack[:5]],
            "Client-facing value stack: this is where the report converts the return into measurable planning themes instead of generic advice.",
            kind="barh",
            width=5.75,
        )


def _strategy_cards(doc, actions, compact=False):
    if not compact:
        _section_title(doc, "Priority Strategy Recommendations", "Client action plan")
    rows_per = 1 if compact else min(len(actions), 6)
    for idx, action in enumerate(actions[: rows_per if compact else 6], start=1):
        table = doc.add_table(rows=1, cols=2)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        _set_widths(table, [0.56, 6.7])
        num = table.cell(0, 0)
        body = table.cell(0, 1)
        num.text = f"{idx}"
        _format_cell(num, fill=BRAND_RED, color=WHITE, bold=True, size=13.5, align=WD_ALIGN_PARAGRAPH.CENTER, border=BRAND_RED)
        _format_cell(body, fill=WHITE, border="DDE3EA")
        title = body.paragraphs[0]
        title.add_run(_safe(action.get("title"), "Planning action"))
        _paragraph(title, size=10.5, color=INK, bold=True, after=2)
        details = body.add_paragraph()
        score = action.get("score")
        if score not in (None, ""):
            details.add_run("Score: ").bold = True
            details.add_run(f"{int(_num(score))}/100   ")
        details.add_run("Impact: ").bold = True
        details.add_run(str(action.get("estimated_savings", "Planning value")))
        details.add_run("   Timing: ").bold = True
        details.add_run(str(action.get("timeline", "Review")))
        _paragraph(details, size=8.4, color=TEXT_GRAY, after=2)
        reason = body.add_paragraph(_limit(action.get("reason", "Client-specific planning opportunity."), 260 if compact else 390))
        _paragraph(reason, size=8.7, color=INK, after=0)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)


def _implementation(doc, actions):
    _section_title(doc, "Implementation Roadmap", "Do this now")
    first = actions[0].get("title", "highest-value planning item") if actions else "highest-value planning item"
    rows = [
        ["Timing", "Client Action", "Advisor Follow-Up"],
        ["Next 30 days", f"Start the first priority: {first}.", "Confirm supporting facts, deadlines, and expected tax impact."],
        ["Next 90 days", "Model retirement, withholding, business, and state planning decisions.", "Prepare implementation projections and client decision points."],
        ["Before year-end", "Execute approved strategies and document the file.", "Review payroll withholding, estimates, charitable credits, and contribution deadlines."],
        ["Annual review", "Update the plan after the next return and income changes.", "Keep the strategy recurring instead of one-time."],
    ]
    _simple_table(doc, rows, [1.05, 3.25, 3.0], font_size=8.1)


def _final_pages(doc, data):
    _section_title(doc, "Final Advisor Recommendation", "Recommendation")
    final = data.get("final_recommendation") or data.get("planning_summary") or "The strongest planning value comes from prioritizing the right strategies in the right order. The client should focus first on actions that improve tax efficiency, reduce compliance risk, improve cash-flow predictability, and support long-term wealth building."
    p = doc.add_paragraph(_limit(final, 760))
    _paragraph(p, size=9.3, color=INK, after=8)

    _callout(doc, "Client Next Step", "Review the priority actions, select the strategies to implement, and schedule a follow-up planning meeting before the next major tax deadline.", fill=LIGHT_GOLD, accent=BRAND_GOLD)

    _section_title(doc, "Prepared By", "Advisor contact")
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_widths(table, [3.35, 3.9])
    left = table.cell(0, 0)
    right = table.cell(0, 1)
    left.text = "Scott Kunz, ChFC, TPCP\nEnrolled Agent and Financial Advisor\nValhalla Tax & Finance LLC"
    right.text = "7055 W Bell Rd, Suite B20, Glendale, AZ 85308\n(623) 887-7921\nskunz@valhallataxservice.com\nwww.valhallataxservice.com"
    _format_cell(left, fill=LIGHT_RED, bold=True, size=8.7)
    _format_cell(right, fill=WHITE, size=8.5)

    _section_title(doc, "Important Planning Notes", "Disclosure")
    disc = doc.add_paragraph("The savings estimates in this report are planning illustrations, not guaranteed outcomes. Actual results depend on final income, filing status, business-use percentages, payroll requirements, documentation, entity costs, state law, and implementation timing. Strategies should be implemented with proper documentation and professional review.")
    _paragraph(disc, size=8.1, color=TEXT_GRAY, after=0)


def _build_strategy_output(data):
    if data.get("priority_actions") or data.get("dynamic_sections"):
        return {"priority_actions": data.get("priority_actions", []), "dynamic_sections": data.get("dynamic_sections", [])}
    if generate_dynamic_tax_strategy is None:
        return {"priority_actions": _default_actions(data), "dynamic_sections": []}
    try:
        result = generate_dynamic_tax_strategy(data)
        if not result.get("priority_actions"):
            result["priority_actions"] = _default_actions(data)
        return result
    except Exception:
        return {"priority_actions": _default_actions(data), "dynamic_sections": []}


def _build_roi_output(data):
    if data.get("roi_strategies"):
        return {"tax_efficiency": {}, "roi_strategies": data.get("roi_strategies", [])}
    if generate_roi_analysis is None:
        return {"tax_efficiency": {}, "roi_strategies": []}
    try:
        return generate_roi_analysis(data)
    except Exception:
        return {"tax_efficiency": {}, "roi_strategies": []}


def _dynamic_sections(doc, sections):
    if not sections:
        return
    _section_title(doc, "Client-Specific Planning Detail", "Strategy modules")
    for item in sections[:6]:
        if isinstance(item, dict):
            heading = item.get("section") or item.get("title") or "Planning Module"
            body = item.get("body") or item.get("summary") or ""
        else:
            heading = "Planning Module"
            body = str(item)
        h = doc.add_paragraph(heading)
        _paragraph(h, size=10.0, color=BRAND_RED, bold=True, after=2)
        b = doc.add_paragraph(_limit(body, 650))
        _paragraph(b, size=8.8, color=INK, after=5)


def generate_valhalla_docx_report(data: dict, output_path: str = "valhalla_premium_report.docx"):
    data = data or {}
    strategy_output = _build_strategy_output(data)
    roi_output = _build_roi_output(data)
    actions = _augment_report_actions(data, _normalize_actions(data, strategy_output, roi_output))
    dynamic_sections = strategy_output.get("dynamic_sections", [])

    doc = Document()
    _configure(doc, data)

    _cover(doc, data, actions)
    _dashboard(doc, data, actions)
    _tax_position(doc, data)
    _business_section(doc, data)
    _roth_strategy_section(doc, data)
    _capital_gain_strategy_section(doc, data)
    _visuals(doc, data, actions)
    doc.add_page_break()
    _strategy_cards(doc, actions, compact=False)
    _dynamic_sections(doc, dynamic_sections)
    _implementation(doc, actions)
    doc.add_page_break()
    _final_pages(doc, data)

    doc.save(output_path)
    return output_path
