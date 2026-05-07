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

BRAND_RED = "981E26"
LIGHT_RED = "F7E9EA"
GOLD = "FFF7DD"


def _money(value):
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"


def _num(value, default=0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _set_cell_border(cell, color="CCCCCC", size="6"):
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


def _format_cell(cell, bold=False, font_size=8.8, color="000000", fill=None):
    if fill:
        _set_cell_shading(cell, fill)
    _set_cell_border(cell)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for paragraph in cell.paragraphs:
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.space_before = Pt(0)
        for run in paragraph.runs:
            run.bold = bold
            run.font.size = Pt(font_size)
            run.font.color.rgb = RGBColor.from_string(color)
            run.font.name = "Georgia"


def _style_paragraph(paragraph, size=9.3, bold=False, color="000000", italic=False, before=0, after=4):
    paragraph.paragraph_format.space_before = Pt(before)
    paragraph.paragraph_format.space_after = Pt(after)
    for run in paragraph.runs:
        run.font.name = "Georgia"
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        run.font.color.rgb = RGBColor.from_string(color)


def _add_section_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text.upper())
    r.bold = True
    r.font.name = "Georgia"
    r.font.size = Pt(13)
    r.font.color.rgb = RGBColor.from_string(BRAND_RED)
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "10")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), BRAND_RED)
    border.append(bottom)
    p._p.get_or_add_pPr().append(border)


def _add_callout(doc, title, body, fill=LIGHT_RED):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    _set_cell_shading(cell, fill)
    _set_cell_border(cell, color="E5C4C7")
    p = cell.paragraphs[0]
    r = p.add_run(title)
    r.bold = True
    r.font.name = "Georgia"
    r.font.size = Pt(10.2)
    r.font.color.rgb = RGBColor.from_string(BRAND_RED)
    p2 = cell.add_paragraph(body)
    _style_paragraph(p2, size=9.0, after=0)
    doc.add_paragraph()


def _add_footer_text(section):
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.text = "Prepared by Scott Kunz, ChFC, TPCP, Enrolled Agent and Financial Advisor | Confidential client planning document"
    _style_paragraph(footer, size=7.5, color="777777")


def _add_title_header(doc, data):
    header_table = doc.add_table(rows=1, cols=2)
    header_table.autofit = False
    header_table.columns[0].width = Inches(7.3)
    header_table.columns[1].width = Inches(2.0)
    left = header_table.cell(0, 0)
    right = header_table.cell(0, 1)
    p = left.paragraphs[0]
    r = p.add_run("VALHALLA TAX SERVICES\n")
    r.bold = True
    r.font.name = "Georgia"
    r.font.size = Pt(19)
    r.font.color.rgb = RGBColor.from_string(BRAND_RED)
    r2 = p.add_run("Comprehensive Tax Strategy Report")
    r2.bold = True
    r2.font.name = "Georgia"
    r2.font.size = Pt(15)
    logo_path = data.get("logo_path", "valhalla_logo.jpg")
    rp = right.paragraphs[0]
    rp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if os.path.exists(logo_path):
        try:
            rp.add_run().add_picture(logo_path, width=Inches(1.35))
        except Exception:
            rp.add_run("VALHALLA")
    else:
        logo = rp.add_run("VALHALLA")
        logo.bold = True
        logo.font.color.rgb = RGBColor.from_string(BRAND_RED)
    for cell in (left, right):
        _set_cell_border(cell, color="FFFFFF", size="0")
    meta = doc.add_paragraph()
    meta.add_run(f"Client: {data.get('client_name', 'Client')}\n").bold = True
    meta.add_run(f"Tax Year: {data.get('tax_year', '2025')}\n")
    meta.add_run("Prepared by: Scott Kunz, ChFC, TPCP, Enrolled Agent and Financial Advisor")
    _style_paragraph(meta, size=10.3, after=8)


def _add_snapshot_box(doc, data):
    _add_section_heading(doc, "Executive Snapshot")
    rows = [
        ["Metric", "Current Position", "Planning Opportunity"],
        ["Federal Income Tax", "$0" if _num(data.get("taxable_income", 0)) == 0 else "Taxable exposure exists", "Focus planning on the highest-impact tax driver"],
        ["Total Tax", _money(data.get("total_tax", 3429)), "Reduce future tax drag as income changes"],
        ["Schedule C Net Profit", _money(data.get("schedule_c_net_profit", 24266)), "Build toward retirement/entity planning trigger points"],
        ["Refund / Balance", _money(data.get("refund", 6731)), "Improve withholding and cash-flow predictability"],
        ["Top Planning Focus", data.get("top_planning_focus", "Tax strategy prioritization"), "Use ranked recommendations and implementation timeline"],
    ]
    table = doc.add_table(rows=len(rows), cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            _format_cell(cell, bold=(r_idx == 0), color=("FFFFFF" if r_idx == 0 else "000000"), fill=(BRAND_RED if r_idx == 0 else "FFFFFF"), font_size=8.8)


def _add_confirmed_tax_table(doc, data):
    rows = [
        ("Filing Status", data.get("filing_status", "Head of Household")),
        ("Dependents", str(data.get("dependents", 2))),
        ("AGI", _money(data.get("agi", 24266))),
        ("Taxable Income", _money(data.get("taxable_income", 0))),
        ("Total Tax", _money(data.get("total_tax", 3429))),
        ("Refund", _money(data.get("refund", 6731))),
    ]
    table = doc.add_table(rows=2, cols=6)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (label, value) in enumerate(rows):
        table.cell(0, idx).text = label
        table.cell(1, idx).text = value
        _format_cell(table.cell(0, idx), bold=True, color="FFFFFF", fill=BRAND_RED, font_size=8.5)
        _format_cell(table.cell(1, idx), fill="FFFFFF", font_size=8.5)


def _add_business_table(doc, data):
    gross = _num(data.get("schedule_c_gross_revenue", 216265))
    net = _num(data.get("schedule_c_net_profit", 24266))
    expenses = gross - net
    margin = (net / gross * 100) if gross else 0
    rows = [
        ["Business Metric", "Amount", "Advisor Read", "Planning Priority"],
        ["Gross Revenue", _money(gross), "Strong activity level" if gross else "Not provided", "Build structure around growth" if gross else "Review business inputs"],
        ["Total Expenses", f"~{_money(expenses)}", "Very high expense ratio" if gross and expenses / gross > .70 else "Expense ratio appears moderate", "Confirm substantiation and business purpose"],
        ["Net Profit", _money(net), f"About {margin:.1f}% margin" if gross else "Not provided", "Improve profitability without losing tax control"],
        ["Contract Labor", _money(data.get("contract_labor", 0)), "Review if material", "Confirm 1099/W-2 classification and documentation"],
    ]
    table = doc.add_table(rows=len(rows), cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            _format_cell(cell, bold=(r_idx == 0), color=("FFFFFF" if r_idx == 0 else "000000"), fill=(BRAND_RED if r_idx == 0 else "FFFFFF"), font_size=8.7)


def _default_priority_actions(data):
    return [
        {"title": "Clean up Schedule C structure and contractor compliance", "estimated_savings": "Risk reduction plus protects major deductions", "timeline": "Now to 90 days", "reason": "Protects the largest deduction categories and reduces audit exposure."},
        {"title": "Open and fund a Solo 401(k) or SEP IRA", "estimated_savings": "Future benefit $4K-$8K+", "timeline": "Now", "reason": "Creates a repeatable wealth-building deduction strategy as profit increases."},
        {"title": "Build S-Corp trigger model for $60K-$80K profit level", "estimated_savings": "$5K-$7.5K annual future savings", "timeline": "Monitor quarterly", "reason": "S-Corp should be timed to profit, not started too early."},
    ]


def _add_top_actions_table(doc, actions):
    actions = actions or []
    if not actions:
        actions = _default_priority_actions({})
    rows = [["Rank", "Recommendation", "Estimated Impact", "Timing", "Why It Matters"]]
    for idx, action in enumerate(actions[:7], start=1):
        rows.append([
            str(idx),
            action.get("title", "Planning action"),
            action.get("estimated_savings", "TBD"),
            action.get("timeline", "Review"),
            action.get("reason", "Client-specific planning opportunity."),
        ])
    table = doc.add_table(rows=len(rows), cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            _format_cell(cell, bold=(r_idx == 0), color=("FFFFFF" if r_idx == 0 else "000000"), fill=(BRAND_RED if r_idx == 0 else "FFFFFF"), font_size=8.0)


def _add_strategy_table(doc, actions):
    rows = [["Strategy", "Priority", "Estimated Impact", "Timing"]]
    for action in (actions or [])[:7]:
        rows.append([
            action.get("title", "Strategy"),
            action.get("priority", "Review"),
            action.get("estimated_savings", "TBD"),
            action.get("timeline", "Review"),
        ])
    if len(rows) == 1:
        rows.extend([
            ["S-Corp", "Monitor", "$5,000-$7,500 annually", "Future"],
            ["Retirement Plan", "High", "$4,000-$8,000 annually", "Current year"],
            ["Vehicle Strategy", "Review", "$12,000-$18,000 year one", "If fact pattern supports it"],
        ])
    table = doc.add_table(rows=len(rows), cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            _format_cell(cell, bold=(r_idx == 0), color=("FFFFFF" if r_idx == 0 else "000000"), fill=(BRAND_RED if r_idx == 0 else "FFFFFF"), font_size=8.5)


def _create_chart(path, title, labels, values):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.4, 2.15))
        plt.bar(labels, values)
        plt.title(title, fontsize=10)
        plt.tick_params(axis="both", labelsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return True
    except Exception:
        return False


def _add_charts_page(doc, data):
    doc.add_page_break()
    _add_section_heading(doc, "Planning Visuals")
    gross = _num(data.get("schedule_c_gross_revenue", 216265))
    net = _num(data.get("schedule_c_net_profit", 24266))
    expenses = max(gross - net, 0)
    chart_specs = [
        ("Schedule C Revenue, Expenses, and Profit", ["Revenue", "Expenses", "Profit"], [gross, expenses, net]),
        ("Estimated Strategy Savings Ranges", ["S-Corp", "Retirement", "Vehicle", "Child", "QBI"], [7500, 8000, 18000, 8500, 4500]),
    ]
    for title, labels, values in chart_specs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            chart_path = tmp.name
        if _create_chart(chart_path, title, labels, values):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(8)
            p.add_run().add_picture(chart_path, width=Inches(6.15))
        try:
            os.remove(chart_path)
        except Exception:
            pass
    doc.add_page_break()


def _add_dynamic_sections(doc, dynamic_sections):
    if not dynamic_sections:
        return
    _add_section_heading(doc, "Client-Specific Strategy Modules")
    for item in dynamic_sections:
        title = item.get("section", "Strategy Module")
        body = item.get("body", "")
        p = doc.add_paragraph(title)
        _style_paragraph(p, size=10, bold=True, color=BRAND_RED, after=2)
        p = doc.add_paragraph(body)
        _style_paragraph(p, size=9.1)


def _add_signature_block(doc):
    _add_section_heading(doc, "Prepared By")
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    left = table.cell(0, 0)
    right = table.cell(0, 1)
    left.text = "Scott Kunz, ChFC, TPCP\nEnrolled Agent and Financial Advisor\nValhalla Tax Services"
    right.text = "7055 W Bell Rd, Suite B20, Glendale, AZ 85308\n(623) 887-7921\nskunz@valhallataxservice.com\nwww.valhallataxservice.com"
    _format_cell(left, fill=LIGHT_RED, bold=True)
    _format_cell(right, fill="FFFFFF")


def _add_disclaimer(doc):
    _add_section_heading(doc, "Important Planning Notes")
    p = doc.add_paragraph("The savings estimates in this report are planning illustrations, not guaranteed outcomes. Actual results depend on final income, filing status, business-use percentages, payroll requirements, documentation, entity costs, state law, and implementation timing. Strategies involving children, contractors, retirement plans, vehicle deductions, and S-Corp elections should be implemented with proper documentation and professional review.")
    _style_paragraph(p, size=8.5, color="555555")


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


def generate_valhalla_docx_report(data: dict, output_path: str = "valhalla_premium_report.docx"):
    data = data or {}
    strategy_output = _build_strategy_output(data)
    priority_actions = strategy_output.get("priority_actions", [])
    dynamic_sections = strategy_output.get("dynamic_sections", [])

    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.top_margin = Inches(0.45)
    section.bottom_margin = Inches(0.6)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)
    _add_footer_text(section)
    doc.styles["Normal"].font.name = "Georgia"
    doc.styles["Normal"].font.size = Pt(9.3)

    _add_title_header(doc, data)
    _add_callout(doc, "Advisor Summary", "This plan identifies the current tax position, business deduction quality, tax savings opportunities, and the recommended implementation path. The report now uses dynamic strategy logic to prioritize recommendations based on the client facts supplied.")
    _add_callout(doc, "Potential Future Annual Tax Savings Identified", "$15,000-$30,000+ depending on income growth, documentation quality, entity timing, retirement funding, vehicle strategy, and implementation discipline.", fill=GOLD)
    _add_snapshot_box(doc, data)

    _add_section_heading(doc, "Confirmed Tax Position")
    _add_confirmed_tax_table(doc, data)
    p = doc.add_paragraph("Source reviewed: filed income tax return and supplied planning facts. Amounts should be verified against final filed copies before implementation.")
    _style_paragraph(p, size=8.3, color="555555", italic=True)

    _add_section_heading(doc, "Executive Summary")
    summary_items = [
        "The planning engine identifies which strategies are most relevant based on the supplied client fact pattern.",
        "Priority actions are ranked based on estimated impact, timing, and implementation urgency.",
        "The report is designed to move from tax preparation facts to proactive advisory recommendations.",
        "The highest-value planning opportunities are highlighted first so the client knows what to act on.",
    ]
    for item in summary_items:
        p = doc.add_paragraph("- " + item)
        _style_paragraph(p, size=9.2)

    _add_callout(doc, "Primary Planning Message", "The objective is not merely to identify deductions. The objective is to prioritize the strategies that produce the highest planning value and convert them into a clear implementation path.")

    _add_section_heading(doc, "Current Federal Tax Analysis")
    p = doc.add_paragraph("Tax Burden Analysis")
    _style_paragraph(p, size=9.8, bold=True)
    for item in [
        f"Adjusted gross income: {_money(data.get('agi', 0))}.",
        f"Taxable income: {_money(data.get('taxable_income', 0))}.",
        f"Total tax: {_money(data.get('total_tax', 0))}.",
        "Planning focus should be directed toward the highest-impact tax driver rather than generic deductions.",
    ]:
        p = doc.add_paragraph("- " + item)
        _style_paragraph(p, size=9.2)

    if _num(data.get("schedule_c_net_profit", 0)) > 0 or _num(data.get("schedule_c_gross_revenue", 0)) > 0:
        _add_section_heading(doc, "Schedule C Business Analysis")
        _add_business_table(doc, data)
        _add_callout(doc, "Schedule C Risk Point", "Business-owner planning should focus on documentation, entity timing, retirement plan integration, and self-employment tax management.", fill=GOLD)

    _add_charts_page(doc, data)

    _add_section_heading(doc, "Top Priority Actions")
    _add_top_actions_table(doc, priority_actions)
    doc.add_page_break()

    _add_section_heading(doc, "Tax Savings Scorecard")
    _add_strategy_table(doc, priority_actions)

    _add_dynamic_sections(doc, dynamic_sections)

    _add_section_heading(doc, "Do This Now Checklist and Implementation Roadmap")
    roadmap = doc.add_table(rows=4, cols=3)
    roadmap.alignment = WD_TABLE_ALIGNMENT.CENTER
    header = ["Timeline", "Action Items", "Purpose"]
    rows = [
        ["Next 30 Days", "Address the highest-ranked action items and gather supporting documentation.", "Create immediate momentum and reduce implementation risk."],
        ["Next 90 Days", "Model larger strategies such as entity structure, retirement contributions, withholding, and investment tax planning.", "Convert recommendations into measurable planning decisions."],
        ["Next 12 Months", "Review progress quarterly and update the strategy as income, deductions, and family facts change.", "Turn the report into a recurring advisory process."],
    ]
    for i, val in enumerate(header):
        roadmap.cell(0, i).text = val
        _format_cell(roadmap.cell(0, i), bold=True, color="FFFFFF", fill=BRAND_RED)
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, val in enumerate(row):
            roadmap.cell(r_idx, c_idx).text = val
            _format_cell(roadmap.cell(r_idx, c_idx), fill="FFFFFF")
    _add_callout(doc, "Do This Now - Advisor Directive", "Start with the highest-ranked recommendations. The purpose of this report is to convert tax data into an actionable implementation plan, not simply summarize the return.", fill=GOLD)

    _add_section_heading(doc, "Final Advisor Recommendation")
    p = doc.add_paragraph("The strongest planning value comes from prioritizing the right strategies in the right order. This report identifies the actions most likely to improve tax efficiency, reduce compliance risk, improve cash-flow predictability, and support long-term wealth building.")
    _style_paragraph(p, size=9.2)
    _add_signature_block(doc)
    _add_disclaimer(doc)
    doc.save(output_path)
    return output_path


def demo_generate_valhalla_docx():
    sample = {
        "client_name": "Nathan Deratany",
        "tax_year": 2025,
        "filing_status": "Head of Household",
        "dependents": 2,
        "agi": 125000,
        "taxable_income": 92000,
        "total_tax": 14000,
        "refund": 1200,
        "schedule_c_gross_revenue": 216265,
        "schedule_c_net_profit": 85000,
        "contract_labor": 107920,
        "capital_gains": 12000,
        "has_retirement_accounts": True,
        "age": 64,
        "logo_path": "valhalla_logo.jpg",
    }
    return generate_valhalla_docx_report(sample)
