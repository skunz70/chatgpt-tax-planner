from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
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
BRAND_DARK = "2B2B2B"
BRAND_GOLD = "C9A24A"
LIGHT_RED = "F7E9EA"
LIGHT_GOLD = "FFF7DD"
LIGHT_GRAY = "F4F5F7"
MID_GRAY = "D9DDE3"
TEXT_GRAY = "555555"
WHITE = "FFFFFF"


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


def _safe_text(value, fallback="Not provided"):
    if value is None or value == "":
        return fallback
    return str(value)


def _set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def _set_cell_border(cell, color="D9DDE3", size="5"):
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


def _set_cell_margins(cell, top=100, start=120, bottom=100, end=120):
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


def _remove_cell_borders(cell):
    _set_cell_border(cell, color=WHITE, size="0")


def _format_cell(cell, bold=False, font_size=8.8, color="000000", fill=None, align=None):
    if fill:
        _set_cell_shading(cell, fill)
    _set_cell_border(cell)
    _set_cell_margins(cell)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for paragraph in cell.paragraphs:
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.space_before = Pt(0)
        if align is not None:
            paragraph.alignment = align
        for run in paragraph.runs:
            run.bold = bold
            run.font.size = Pt(font_size)
            run.font.color.rgb = RGBColor.from_string(color)
            run.font.name = "Aptos"


def _style_paragraph(paragraph, size=9.5, bold=False, color="000000", italic=False, before=0, after=5):
    paragraph.paragraph_format.space_before = Pt(before)
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.08
    for run in paragraph.runs:
        run.font.name = "Aptos"
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        run.font.color.rgb = RGBColor.from_string(color)


def _set_table_widths(table, widths):
    table.autofit = False
    for row in table.rows:
        for idx, width in enumerate(widths):
            if idx < len(row.cells):
                row.cells[idx].width = Inches(width)


def _add_page_number(paragraph):
    run = paragraph.add_run()
    fld_char_1 = OxmlElement("w:fldChar")
    fld_char_1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_char_2 = OxmlElement("w:fldChar")
    fld_char_2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char_1)
    run._r.append(instr)
    run._r.append(fld_char_2)


def _add_footer(section):
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Valhalla Tax Services | Confidential client planning report | Page ")
    _add_page_number(footer)
    _style_paragraph(footer, size=7.4, color="777777", after=0)


def _add_header(section, data):
    header = section.header.paragraphs[0]
    header.text = f"Valhalla Premium Tax Strategy Report | {_safe_text(data.get('client_name'), 'Client')} | {_safe_text(data.get('tax_year'), 'Tax Year')}"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _style_paragraph(header, size=7.8, color="777777", after=0)


def _configure_document(doc, data):
    section = doc.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.62)
    section.right_margin = Inches(0.62)
    _add_header(section, data)
    _add_footer(section)

    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(9.5)
    for style_name in ("Heading 1", "Heading 2", "Heading 3"):
        if style_name in styles:
            styles[style_name].font.name = "Aptos"
            styles[style_name].font.color.rgb = RGBColor.from_string(BRAND_RED)


def _add_section_heading(doc, text, kicker=None):
    if kicker:
        k = doc.add_paragraph(kicker.upper())
        _style_paragraph(k, size=7.6, bold=True, color=BRAND_GOLD, before=6, after=0)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    r.bold = True
    r.font.name = "Aptos Display"
    r.font.size = Pt(15)
    r.font.color.rgb = RGBColor.from_string(BRAND_RED)
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "7")
    bottom.set(qn("w:space"), "2")
    bottom.set(qn("w:color"), MID_GRAY)
    border.append(bottom)
    p._p.get_or_add_pPr().append(border)


def _add_small_label(paragraph, label, value, value_color=BRAND_DARK):
    r = paragraph.add_run(label.upper() + "\n")
    r.bold = True
    r.font.name = "Aptos"
    r.font.size = Pt(7.2)
    r.font.color.rgb = RGBColor.from_string(TEXT_GRAY)
    r2 = paragraph.add_run(str(value))
    r2.bold = True
    r2.font.name = "Aptos Display"
    r2.font.size = Pt(12.5)
    r2.font.color.rgb = RGBColor.from_string(value_color)


def _add_cover_page(doc, data):
    top = doc.add_table(rows=1, cols=2)
    top.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_widths(top, [5.15, 2.0])
    left = top.cell(0, 0)
    right = top.cell(0, 1)
    for cell in (left, right):
        _remove_cell_borders(cell)
        _set_cell_margins(cell, top=60, bottom=60)

    brand = left.paragraphs[0]
    brand.add_run("VALHALLA TAX SERVICES")
    _style_paragraph(brand, size=11, bold=True, color=BRAND_RED, after=0)
    sub = left.add_paragraph("Tax planning | Advisory | Implementation roadmap")
    _style_paragraph(sub, size=8.5, color=TEXT_GRAY, after=0)

    logo_path = data.get("logo_path", "valhalla_logo.jpg")
    rp = right.paragraphs[0]
    rp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if os.path.exists(logo_path):
        try:
            rp.add_run().add_picture(logo_path, width=Inches(1.3))
        except Exception:
            rp.add_run("VALHALLA")
    else:
        logo = rp.add_run("VALHALLA")
        logo.bold = True
        logo.font.color.rgb = RGBColor.from_string(BRAND_RED)
        logo.font.size = Pt(12)

    doc.add_paragraph()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.add_run("Premium Tax\nStrategy Report")
    _style_paragraph(title, size=28, bold=True, color=BRAND_DARK, before=22, after=6)

    subtitle = doc.add_paragraph("Client-ready planning summary, strategy priorities, and implementation roadmap")
    _style_paragraph(subtitle, size=11.5, color=TEXT_GRAY, after=16)

    meta = doc.add_table(rows=1, cols=3)
    meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_widths(meta, [2.25, 2.25, 2.25])
    meta_values = [
        ("Prepared for", _safe_text(data.get("client_name"), "Client")),
        ("Tax year", _safe_text(data.get("tax_year"), "2025")),
        ("Prepared by", "Scott Kunz, ChFC, TPCP, EA"),
    ]
    for idx, (label, value) in enumerate(meta_values):
        cell = meta.cell(0, idx)
        _format_cell(cell, fill=LIGHT_GRAY, color=BRAND_DARK, font_size=9)
        _add_small_label(cell.paragraphs[0], label, value, BRAND_RED if idx == 0 else BRAND_DARK)

    doc.add_paragraph()
    summary = data.get("advisor_summary") or "This report converts the reviewed tax return and supplied planning facts into a prioritized advisory roadmap. It is designed to help the client understand the highest-value opportunities, why they matter, and what to implement next."
    _add_callout(doc, "Advisor Summary", summary, fill=LIGHT_RED, accent=BRAND_RED)

    contents = doc.add_table(rows=4, cols=2)
    contents.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_widths(contents, [0.55, 6.35])
    items = [
        ("01", "Executive dashboard and confirmed tax position"),
        ("02", "Tax visuals and business-owner analysis"),
        ("03", "Top strategy priorities and ROI scorecard"),
        ("04", "Implementation roadmap and advisor recommendation"),
    ]
    for row, (num, text) in zip(contents.rows, items):
        row.cells[0].text = num
        row.cells[1].text = text
        _format_cell(row.cells[0], bold=True, color=WHITE, fill=BRAND_RED, align=WD_ALIGN_PARAGRAPH.CENTER)
        _format_cell(row.cells[1], fill=WHITE, font_size=9.2)
    doc.add_page_break()


def _add_callout(doc, title, body, fill=LIGHT_RED, accent=BRAND_RED):
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_widths(table, [0.16, 6.95])
    bar = table.cell(0, 0)
    cell = table.cell(0, 1)
    _set_cell_shading(bar, accent)
    _set_cell_border(bar, color=accent)
    _set_cell_margins(bar, top=60, bottom=60, start=20, end=20)
    _set_cell_shading(cell, fill)
    _set_cell_border(cell, color="E8D4D6")
    _set_cell_margins(cell, top=115, start=145, bottom=115, end=145)
    p = cell.paragraphs[0]
    p.add_run(title)
    _style_paragraph(p, size=9.7, bold=True, color=accent, after=2)
    p2 = cell.add_paragraph(body)
    _style_paragraph(p2, size=9.0, color=BRAND_DARK, after=0)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(3)


def _refund_or_due(data):
    balance = _num(data.get("balance_due", 0))
    refund = _num(data.get("refund", 0))
    if balance > 0:
        return _money(-balance), "Balance due"
    if refund > 0:
        return _money(refund), "Refund"
    return "$0", "Refund / balance"


def _add_metric_cards(doc, data):
    value, label = _refund_or_due(data)
    cards = [
        ("AGI", _money(data.get("agi", 0)), "Income baseline", BRAND_RED),
        ("Taxable income", _money(data.get("taxable_income", 0)), "Bracket planning base", BRAND_DARK),
        ("Total tax", _money(data.get("total_tax", 0)), "Current federal burden", BRAND_DARK),
        (label, value, "Cash-flow result", BRAND_GOLD if value.startswith("(") else BRAND_RED),
    ]
    table = doc.add_table(rows=1, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_widths(table, [1.72, 1.72, 1.72, 1.72])
    for idx, (label, amount, note, color) in enumerate(cards):
        cell = table.cell(0, idx)
        _format_cell(cell, fill=LIGHT_GRAY, font_size=8.6)
        p = cell.paragraphs[0]
        _add_small_label(p, label, amount, color)
        p2 = cell.add_paragraph(note)
        _style_paragraph(p2, size=7.6, color=TEXT_GRAY, after=0)


def _add_dashboard(doc, data):
    _add_section_heading(doc, "Executive Dashboard", "Client snapshot")
    _add_metric_cards(doc, data)
    doc.add_paragraph()

    _add_callout(
        doc,
        "Primary Planning Message",
        data.get("top_planning_focus") or "Prioritize the strategies that produce the highest planning value first, then convert them into a clear implementation path with dates, owner, and documentation requirements.",
        fill=LIGHT_GOLD,
        accent=BRAND_GOLD,
    )

    rows = [
        ["Planning Area", "Current Read", "Advisor Focus"],
        ["Federal tax position", f"Taxable income of {_money(data.get('taxable_income', 0))} with total tax of {_money(data.get('total_tax', 0))}", "Bracket management, withholding, and timing decisions"],
        ["Cash flow", f"{_refund_or_due(data)[1]} of {_refund_or_due(data)[0]}", "Improve predictability before year-end"],
        ["Business activity", f"Schedule C net profit of {_money(data.get('schedule_c_net_profit', 0))}", "Documentation, retirement plan options, and entity trigger review"],
        ["Investment income", f"Capital gains {_money(data.get('capital_gains', 0))}; dividends {_money(data.get('dividend_income', 0))}", "Harvesting, asset location, and future taxable-income control"],
    ]
    _add_table(doc, rows, widths=[1.55, 2.75, 2.9], header_fill=BRAND_RED, body_fill=WHITE, font_size=8.5)


def _add_table(doc, rows, widths, header_fill=BRAND_RED, body_fill=WHITE, font_size=8.5):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_widths(table, widths)
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = str(val)
            is_header = r_idx == 0
            fill = header_fill if is_header else (LIGHT_GRAY if r_idx % 2 == 0 else body_fill)
            align = WD_ALIGN_PARAGRAPH.CENTER if c_idx == 0 or (is_header and len(str(val)) < 18) else WD_ALIGN_PARAGRAPH.LEFT
            _format_cell(
                cell,
                bold=is_header,
                color=WHITE if is_header else BRAND_DARK,
                fill=fill,
                font_size=font_size if not is_header else font_size + 0.2,
                align=align,
            )
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(5)
    return table


def _add_confirmed_tax_position(doc, data):
    _add_section_heading(doc, "Confirmed Tax Position", "Return facts")
    status = _safe_text(data.get("filing_status"), "Not provided")
    rows = [
        ["Item", "Amount / Status", "Planning Note"],
        ["Filing status", status, "Use the correct bracket, standard deduction, and phaseout assumptions"],
        ["Dependents", _safe_text(data.get("dependents"), "0"), "Relevant for credits, education planning, and family payroll analysis"],
        ["AGI", _money(data.get("agi", 0)), "Baseline for phaseouts, credits, IRMAA, and bracket strategy"],
        ["Taxable income", _money(data.get("taxable_income", 0)), "Main driver for Roth, gain harvesting, and contribution planning"],
        ["Total tax", _money(data.get("total_tax", 0)), "Current tax-cost benchmark"],
        [_refund_or_due(data)[1], _refund_or_due(data)[0], "Withholding and estimate planning should align with future projections"],
    ]
    _add_table(doc, rows, widths=[1.55, 2.05, 3.6])
    p = doc.add_paragraph("Source reviewed: filed income tax return and supplied planning facts. Amounts should be verified against final filed copies before implementation.")
    _style_paragraph(p, size=8.0, color=TEXT_GRAY, italic=True, after=4)


def _default_priority_actions(data):
    return [
        {"title": "Clean up Schedule C structure and contractor compliance", "estimated_savings": "Risk reduction plus protects major deductions", "timeline": "Now to 90 days", "reason": "Protects the largest deduction categories and reduces audit exposure."},
        {"title": "Open and fund a Solo 401(k) or SEP IRA", "estimated_savings": "Future benefit $4K-$8K+", "timeline": "Now", "reason": "Creates a repeatable wealth-building deduction strategy as profit increases."},
        {"title": "Build S-Corp trigger model for $60K-$80K profit level", "estimated_savings": "$5K-$7.5K annual future savings", "timeline": "Monitor quarterly", "reason": "S-Corp should be timed to profit, not started too early."},
    ]


def _merge_roi_into_actions(priority_actions, roi_strategies):
    actions = list(priority_actions or [])
    for roi in roi_strategies or []:
        actions.append({
            "priority": roi.get("priority", "Review"),
            "title": roi.get("strategy", "ROI Strategy"),
            "estimated_savings": _money(roi.get("estimated_savings", 0)),
            "timeline": roi.get("timeline", "Review"),
            "reason": roi.get("advisor_reasoning", "Quantified planning opportunity identified."),
            "score": roi.get("score", ""),
        })
    actions.sort(key=lambda x: _num(x.get("score", 0)), reverse=True)
    return actions[:7]


def _create_chart(path, title, labels, values, kind="barh"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        values = [_num(v) for v in values]
        colors = ["#981E26", "#C9A24A", "#555555", "#8FA3B6", "#D9DDE3", "#6E7F8F"]
        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(6.9, 2.95))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        if kind == "barh":
            order = list(range(len(labels)))[::-1]
            ax.barh([labels[i] for i in order], [values[i] for i in order], color=[colors[i % len(colors)] for i in order], height=0.55)
            ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
            ax.tick_params(axis="x", labelsize=7.5, colors="#555555")
            ax.tick_params(axis="y", labelsize=8.2, colors="#2B2B2B")
            ax.grid(axis="x", color="#E6E8EB", linewidth=0.8)
        elif kind == "donut":
            total = sum(max(v, 0) for v in values) or 1
            safe_values = [max(v, 0) for v in values]
            ax.pie(safe_values, labels=labels, colors=colors[:len(labels)], startangle=90, wedgeprops={"width": 0.42, "edgecolor": "white"}, textprops={"fontsize": 8})
            ax.text(0, 0, _money(total), ha="center", va="center", fontsize=12, fontweight="bold", color="#981E26")
        else:
            ax.bar(labels, values, color=colors[:len(labels)], width=0.55)
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
            ax.tick_params(axis="x", labelsize=7.5, rotation=15, colors="#2B2B2B")
            ax.tick_params(axis="y", labelsize=7.5, colors="#555555")
            ax.grid(axis="y", color="#E6E8EB", linewidth=0.8)

        ax.set_title(title, fontsize=10.2, fontweight="bold", color="#2B2B2B", pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=1.0)
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        return False


def _add_chart_image(doc, title, labels, values, kind="barh"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        chart_path = tmp.name
    try:
        if _create_chart(chart_path, title, labels, values, kind=kind):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(8)
            p.add_run().add_picture(chart_path, width=Inches(6.55))
            return True
    finally:
        try:
            os.remove(chart_path)
        except Exception:
            pass
    return False


def _add_visuals(doc, data, roi_strategies):
    doc.add_page_break()
    _add_section_heading(doc, "Planning Visuals", "Charts")
    _add_chart_image(
        doc,
        "Federal Tax Position Snapshot",
        ["AGI", "Taxable Income", "Total Tax", "Withholding", _refund_or_due(data)[1]],
        [data.get("agi", 0), data.get("taxable_income", 0), data.get("total_tax", 0), data.get("federal_withholding", 0), abs(_num(data.get("balance_due", 0)) or _num(data.get("refund", 0)))],
        kind="barh",
    )

    gross = _num(data.get("schedule_c_gross_revenue", 0))
    net = _num(data.get("schedule_c_net_profit", 0))
    expenses = max(gross - net, 0)
    if gross or net:
        _add_chart_image(
            doc,
            "Schedule C Economics",
            ["Expenses", "Net Profit"],
            [expenses, net],
            kind="donut",
        )

    if roi_strategies:
        _add_chart_image(
            doc,
            "ROI-Ranked Estimated Strategy Savings",
            [str(s.get("strategy", "Strategy"))[:22] for s in roi_strategies[:5]],
            [_num(s.get("estimated_savings", 0)) for s in roi_strategies[:5]],
            kind="bar",
        )
    else:
        _add_chart_image(
            doc,
            "Illustrative Strategy Savings Ranges",
            ["Retirement", "Entity", "Withholding", "Investments", "Credits"],
            [8000, 7500, 3500, 4500, 2500],
            kind="bar",
        )


def _add_business_analysis(doc, data):
    gross = _num(data.get("schedule_c_gross_revenue", 0))
    net = _num(data.get("schedule_c_net_profit", 0))
    if not gross and not net:
        return
    expenses = max(gross - net, 0)
    _add_section_heading(doc, "Business-Owner Analysis", "Schedule C")
    rows = [
        ["Metric", "Amount", "Advisor Read"],
        ["Gross revenue", _money(gross), "Business activity is material enough to warrant proactive structure review" if gross else "Not provided"],
        ["Expenses", _money(expenses), f"Expense ratio: {_pct(expenses, gross)}" if gross else "Confirm complete expense detail"],
        ["Net profit", _money(net), f"Net margin: {_pct(net, gross)}" if gross else "Retirement plan and entity timing depend on profit"],
        ["Contract labor", _money(data.get("contract_labor", 0)), "Confirm classification, W-9 files, and 1099 documentation"],
    ]
    _add_table(doc, rows, widths=[1.55, 1.65, 3.95])
    _add_callout(doc, "Schedule C Risk Point", "Business-owner planning should focus on documentation, entity timing, retirement plan integration, and self-employment tax management.", fill=LIGHT_GOLD, accent=BRAND_GOLD)


def _add_action_card(doc, number, action):
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_widths(table, [0.55, 6.65])
    num = table.cell(0, 0)
    body = table.cell(0, 1)
    num.text = str(number)
    _format_cell(num, bold=True, color=WHITE, fill=BRAND_RED, font_size=13, align=WD_ALIGN_PARAGRAPH.CENTER)
    _format_cell(body, fill=WHITE, font_size=8.8)
    title = body.paragraphs[0]
    title.add_run(action.get("title", "Planning action"))
    _style_paragraph(title, size=10.4, bold=True, color=BRAND_DARK, after=2)
    detail = body.add_paragraph()
    detail.add_run("Impact: ").bold = True
    detail.add_run(action.get("estimated_savings", "TBD"))
    detail.add_run("   |   Timing: ").bold = True
    detail.add_run(action.get("timeline", "Review"))
    _style_paragraph(detail, size=8.6, color=TEXT_GRAY, after=2)
    reason = body.add_paragraph(action.get("reason", "Client-specific planning opportunity."))
    _style_paragraph(reason, size=8.7, color=BRAND_DARK, after=0)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)


def _add_priority_actions(doc, actions):
    _add_section_heading(doc, "Top Priority Actions", "Implementation priorities")
    for idx, action in enumerate((actions or _default_priority_actions({}))[:5], start=1):
        _add_action_card(doc, idx, action)


def _add_roi_scorecard(doc, roi_strategies):
    _add_section_heading(doc, "ROI-Ranked Strategy Scorecard", "Quantified opportunities")
    rows = [["Rank", "Strategy", "Savings", "Score", "Difficulty", "Timeline"]]
    for idx, item in enumerate((roi_strategies or [])[:7], start=1):
        rows.append([
            str(idx),
            item.get("strategy", "Strategy"),
            _money(item.get("estimated_savings", 0)),
            f"{item.get('score', 'N/A')}/100",
            item.get("implementation_difficulty", "Review"),
            item.get("timeline", "Review"),
        ])
    if len(rows) == 1:
        rows.append(["1", "No quantified strategy available", "$0", "N/A", "Review", "Review"])
    _add_table(doc, rows, widths=[0.48, 2.35, 1.0, 0.82, 1.1, 1.25], font_size=7.9)


def _add_dynamic_sections(doc, dynamic_sections):
    if not dynamic_sections:
        return
    _add_section_heading(doc, "Client-Specific Strategy Modules", "Planning detail")
    for item in dynamic_sections:
        p = doc.add_paragraph(item.get("section", "Strategy Module"))
        _style_paragraph(p, size=10.2, bold=True, color=BRAND_RED, after=2)
        p = doc.add_paragraph(item.get("body", ""))
        _style_paragraph(p, size=9.0, color=BRAND_DARK, after=5)


def _add_roi_commentary(doc, roi_strategies):
    if not roi_strategies:
        return
    _add_section_heading(doc, "Advisor ROI Commentary", "Why these rank first")
    for item in roi_strategies[:5]:
        p = doc.add_paragraph(item.get("strategy", "Strategy"))
        _style_paragraph(p, size=10.1, bold=True, color=BRAND_RED, after=2)
        p = doc.add_paragraph(item.get("advisor_reasoning", ""))
        _style_paragraph(p, size=9.0, color=BRAND_DARK, after=5)


def _add_roadmap(doc):
    _add_section_heading(doc, "Implementation Roadmap", "Do this now")
    rows = [
        ["Timeline", "Action Items", "Purpose", "Advisor Follow-Up"],
        ["Next 30 days", "Address the highest-ranked recommendations and gather supporting documentation.", "Create immediate momentum and reduce implementation risk.", "Confirm documents, income inputs, and year-end deadlines."],
        ["Next 90 days", "Model entity structure, retirement contributions, withholding, and investment tax planning.", "Convert recommendations into measurable planning decisions.", "Prepare strategy-specific projections."],
        ["Before year-end", "Execute approved moves and document each strategy before deadlines.", "Capture the tax benefit while keeping compliance clean.", "Review withholding, estimates, and final transactions."],
        ["Annual review", "Update the plan as income, deductions, family facts, and law changes evolve.", "Make tax planning a recurring advisory process.", "Refresh plan annually after return review."],
    ]
    _add_table(doc, rows, widths=[1.0, 2.45, 1.95, 1.8], font_size=7.9)
    _add_callout(doc, "Advisor Directive", "Start with the highest-ranked recommendations. The purpose of this report is to convert tax data into an actionable implementation plan, not simply summarize the return.", fill=LIGHT_GOLD, accent=BRAND_GOLD)


def _add_signature_block(doc):
    _add_section_heading(doc, "Prepared By", "Advisor contact")
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_widths(table, [3.35, 3.85])
    left = table.cell(0, 0)
    right = table.cell(0, 1)
    left.text = "Scott Kunz, ChFC, TPCP\nEnrolled Agent and Financial Advisor\nValhalla Tax Services"
    right.text = "7055 W Bell Rd, Suite B20, Glendale, AZ 85308\n(623) 887-7921\nskunz@valhallataxservice.com\nwww.valhallataxservice.com"
    _format_cell(left, fill=LIGHT_RED, bold=True, font_size=8.9)
    _format_cell(right, fill=WHITE, font_size=8.7)


def _add_disclaimer(doc):
    _add_section_heading(doc, "Important Planning Notes", "Disclosure")
    p = doc.add_paragraph("The savings estimates in this report are planning illustrations, not guaranteed outcomes. Actual results depend on final income, filing status, business-use percentages, payroll requirements, documentation, entity costs, state law, and implementation timing. Strategies involving children, contractors, retirement plans, vehicle deductions, and S-Corp elections should be implemented with proper documentation and professional review.")
    _style_paragraph(p, size=8.2, color=TEXT_GRAY, after=0)


def _build_strategy_output(data):
    if generate_dynamic_tax_strategy is None:
        return {"priority_actions": _default_priority_actions(data), "dynamic_sections": []}
    try:
        result = generate_dynamic_tax_strategy(data)
        if not result.get("priority_actions"):
            result["priority_actions"] = _default_priority_actions(data)
        return result
    except Exception:
        return {"priority_actions": _default_priority_actions(data), "dynamic_sections": []}


def _build_roi_output(data):
    if generate_roi_analysis is None:
        return {"tax_efficiency": {}, "roi_strategies": []}
    try:
        return generate_roi_analysis(data)
    except Exception:
        return {"tax_efficiency": {}, "roi_strategies": []}


def generate_valhalla_docx_report(data: dict, output_path: str = "valhalla_premium_report.docx"):
    data = data or {}
    strategy_output = _build_strategy_output(data)
    roi_output = _build_roi_output(data)
    roi_strategies = roi_output.get("roi_strategies", [])
    priority_actions = _merge_roi_into_actions(strategy_output.get("priority_actions", []), roi_strategies)
    dynamic_sections = strategy_output.get("dynamic_sections", [])

    doc = Document()
    _configure_document(doc, data)

    _add_cover_page(doc, data)
    _add_dashboard(doc, data)
    _add_confirmed_tax_position(doc, data)
    _add_business_analysis(doc, data)
    _add_visuals(doc, data, roi_strategies)
    _add_priority_actions(doc, priority_actions)
    _add_roi_scorecard(doc, roi_strategies)
    _add_dynamic_sections(doc, dynamic_sections)
    _add_roi_commentary(doc, roi_strategies)
    _add_roadmap(doc)

    _add_section_heading(doc, "Final Advisor Recommendation", "Recommendation")
    p = doc.add_paragraph("The strongest planning value comes from prioritizing the right strategies in the right order. This report identifies the actions most likely to improve tax efficiency, reduce compliance risk, improve cash-flow predictability, and support long-term wealth building.")
    _style_paragraph(p, size=9.3, color=BRAND_DARK, after=6)
    _add_signature_block(doc)
    _add_disclaimer(doc)

    doc.save(output_path)
    return output_path
