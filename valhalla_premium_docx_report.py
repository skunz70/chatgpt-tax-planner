from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import os

BRAND_RED = "981E26"
DARK_GRAY = "444444"
LIGHT_RED = "F7E9EA"
LIGHT_GRAY = "F2F2F2"


def _money(value):
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"


def _set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _set_cell_border(cell, color="CCCCCC", size="6"):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def _format_cell(cell, bold=False, font_size=9.5, color="000000", fill=None):
    if fill:
        _set_cell_shading(cell, fill)
    _set_cell_border(cell)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for paragraph in cell.paragraphs:
        paragraph.paragraph_format.space_after = Pt(0)
        for run in paragraph.runs:
            run.bold = bold
            run.font.size = Pt(font_size)
            run.font.color.rgb = RGBColor.from_string(color)
            run.font.name = "Georgia"


def _style_paragraph(paragraph, size=10, bold=False, color="000000", italic=False, before=0, after=6):
    paragraph.paragraph_format.space_before = Pt(before)
    paragraph.paragraph_format.space_after = Pt(after)
    for run in paragraph.runs:
        run.font.name = "Georgia"
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        run.font.color.rgb = RGBColor.from_string(color)


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
    r.font.size = Pt(18)
    r.font.color.rgb = RGBColor.from_string(BRAND_RED)
    r2 = p.add_run("Comprehensive Tax Strategy Report")
    r2.bold = True
    r2.font.name = "Georgia"
    r2.font.size = Pt(15)
    r2.font.color.rgb = RGBColor.from_string("000000")

    logo_path = data.get("logo_path", "valhalla_logo.jpg")
    rp = right.paragraphs[0]
    rp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if os.path.exists(logo_path):
        try:
            rp.add_run().add_picture(logo_path, width=Inches(1.35))
        except Exception:
            rp.add_run("VALHALLA")
    else:
        run = rp.add_run("VALHALLA")
        run.bold = True
        run.font.color.rgb = RGBColor.from_string(BRAND_RED)

    for cell in (left, right):
        _set_cell_border(cell, color="FFFFFF", size="0")

    meta = doc.add_paragraph()
    meta.add_run(f"Client: {data.get('client_name', 'Client')}\n").bold = True
    meta.add_run(f"Tax Year: {data.get('tax_year', '2025')}\n")
    meta.add_run("Prepared by: Scott Kunz, ChFC, TPCP, Enrolled Agent and Financial Advisor")
    _style_paragraph(meta, size=10.5, after=8)


def _add_section_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
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
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor.from_string(BRAND_RED)
    p2 = cell.add_paragraph(body)
    _style_paragraph(p2, size=9.5, after=0)
    doc.add_paragraph()


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
        _format_cell(table.cell(0, idx), bold=True, color="FFFFFF", fill=BRAND_RED)
        _format_cell(table.cell(1, idx), bold=False, fill="FFFFFF")


def _add_business_table(doc, data):
    gross = data.get("schedule_c_gross_revenue", 216265)
    net = data.get("schedule_c_net_profit", 24266)
    expenses = float(gross) - float(net)
    rows = [
        ["Business Metric", "Amount", "Advisor Read", "Planning Priority"],
        ["Gross Revenue", _money(gross), "Strong activity level", "Build structure around growth"],
        ["Total Expenses", f"~{_money(expenses)}", "Very high expense ratio", "Confirm substantiation and business purpose"],
        ["Net Profit", _money(net), "About 11% margin", "Improve profitability without losing tax control"],
    ]
    table = doc.add_table(rows=len(rows), cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths = [1.7, 1.2, 2.4, 3.0]
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            cell.width = Inches(widths[c_idx])
            if r_idx == 0:
                _format_cell(cell, bold=True, color="FFFFFF", fill=BRAND_RED)
            else:
                _format_cell(cell, fill="FFFFFF")


def _add_strategy_table(doc):
    rows = [
        ["Strategy", "Current-Year Applicability", "Modeled Scenario", "Estimated Tax Impact"],
        ["S-Corp", "Monitor only", "Profit increases to approx. $80K with reasonable salary planning", "$5,000-$7,500 annually"],
        ["Retirement Plan", "Set up now; savings grows with profit", "Solo 401(k) or SEP IRA funding at higher profit", "$4,000-$8,000 annually"],
        ["Child Employment", "Evaluate facts", "Reasonable wages for documented business work", "$6,000-$8,500 annually"],
        ["Vehicle Strategy", "Compare methods", "Business vehicle with high business use and Section 179/bonus planning", "$12,000-$18,000 year one"],
        ["QBI Optimization", "Already active", "Profit growth with 20% QBI deduction capacity", "$3,500-$4,500"],
    ]
    table = doc.add_table(rows=len(rows), cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            if r_idx == 0:
                _format_cell(cell, bold=True, color="FFFFFF", fill=BRAND_RED)
            else:
                _format_cell(cell, fill="FFFFFF")


def _add_top_actions_table(doc):
    rows = [
        ["Rank", "Recommendation", "Estimated Impact", "Timing", "Why It Matters"],
        ["1", "Clean up Schedule C structure and contractor compliance", "Risk reduction plus protects $100K+ deductions", "Now to 90 days", "This protects the largest deduction category and reduces audit exposure."],
        ["2", "Open and fund a Solo 401(k) or SEP IRA", "Current year benefit may be limited; future benefit $4K-$8K+", "Now", "Creates a repeatable wealth-building deduction strategy as profit increases."],
        ["3", "Build S-Corp trigger model for $60K-$80K profit level", "$5K-$7.5K annual future savings", "Monitor quarterly", "S-Corp should be timed to profit, not started too early."],
    ]
    table = doc.add_table(rows=len(rows), cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            if r_idx == 0:
                _format_cell(cell, bold=True, color="FFFFFF", fill=BRAND_RED, font_size=8.5)
            else:
                _format_cell(cell, fill="FFFFFF", font_size=8.5)


def _add_footer_text(section):
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.text = "Prepared by Scott Kunz, ChFC, TPCP, Enrolled Agent and Financial Advisor | Confidential client planning document"
    _style_paragraph(footer, size=8, color="777777")


def generate_valhalla_docx_report(data: dict, output_path: str = "valhalla_premium_report.docx"):
    data = data or {}
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.top_margin = Inches(0.45)
    section.bottom_margin = Inches(0.45)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)
    _add_footer_text(section)

    styles = doc.styles
    styles["Normal"].font.name = "Georgia"
    styles["Normal"].font.size = Pt(9.5)

    _add_title_header(doc, data)
    _add_callout(
        doc,
        "Advisor Summary",
        "This plan identifies the current tax position, business deduction quality, tax savings opportunities, and the recommended implementation path. The current return shows no federal income tax exposure, but a clear self-employment tax burden and strong opportunity to improve structure as business profit scales.",
    )

    _add_section_heading(doc, "Confirmed Tax Position")
    _add_confirmed_tax_table(doc, data)
    p = doc.add_paragraph("Source reviewed: 2025 filed income tax return, including Form 1040, Schedule C, Schedule SE, QBI schedules, and related supporting schedules. Amounts should be verified against final filed copies before implementation.")
    _style_paragraph(p, size=8.5, color="555555", italic=True)

    _add_section_heading(doc, "Executive Summary")
    summary_items = [
        "The return is currently driven by Schedule C income and refundable family credits.",
        "Federal taxable income is zero after the standard deduction.",
        "The primary tax cost is self-employment tax, not income tax.",
        "The business has strong revenue but a low reported profit margin.",
        "The highest future tax savings come from S-Corp timing, retirement funding, child employment, and vehicle planning.",
    ]
    for item in summary_items:
        doc.add_paragraph(item, style=None).style = doc.styles["Normal"]

    _add_callout(doc, "Primary Planning Message", "The client is not currently paying federal income tax. The planning target is self-employment tax exposure, business structure, cash-flow control, and building tax-efficient wealth as Schedule C profit increases.")

    _add_section_heading(doc, "Current Federal Tax Analysis")
    p = doc.add_paragraph("Tax Burden Analysis")
    _style_paragraph(p, size=10, bold=True)
    for item in [
        "Federal income tax: $0 because taxable income is reduced to $0.",
        "Self-employment tax: $3,429, generated from Schedule C net earnings.",
        "Refund is driven primarily by refundable child credits, not withholding.",
        "This is a favorable current cash-flow result, but it can mask weak estimated tax discipline if profit rises.",
    ]:
        p = doc.add_paragraph("- " + item)
        _style_paragraph(p, size=9.5)

    _add_section_heading(doc, "Schedule C Business Analysis")
    _add_business_table(doc, data)
    _add_callout(doc, "Schedule C Risk Point", "The contract labor amount is the largest compliance item. The recommendation is to document it properly, confirm independent contractor status, issue required Forms 1099, and evaluate whether any workers should be moved to payroll as the business scales.", fill="FFF7DD")

    _add_section_heading(doc, "Top 3 Priority Actions")
    _add_top_actions_table(doc)

    _add_section_heading(doc, "Strategy Impact By Category")
    _add_strategy_table(doc)

    _add_section_heading(doc, "Detailed Strategy Notes")
    notes = [
        ("S-Corporation Timing", "The S-Corp is a future-state strategy, not an automatic current-year recommendation. At $24,266 of profit, administrative costs, payroll, accounting, and reasonable compensation requirements may consume much of the benefit. The correct trigger is to monitor net profit quarterly and revisit the election when consistent annual profit approaches $60,000-$80,000."),
        ("Retirement Plan Funding", "The client should establish a self-employed retirement plan now so the structure is ready before income rises. Because current taxable income is already $0, immediate income tax savings may be limited. The long-term benefit is creating a repeatable deduction and wealth-building mechanism."),
        ("Child Employment Strategy", "If the children perform legitimate work for the business, reasonable wages can shift income from the parent business to the children. This requires timesheets, job descriptions, actual payment, and age-appropriate work."),
        ("Vehicle and Equipment Planning", "Current vehicle deductions appear modest relative to the size of the business. A future vehicle strategy should compare standard mileage, actual expense, depreciation, Section 179, bonus depreciation, business-use percentage, and cash-flow impact."),
    ]
    for title, body in notes:
        p = doc.add_paragraph(title)
        _style_paragraph(p, size=10.5, bold=True, color=BRAND_RED, after=2)
        p = doc.add_paragraph(body)
        _style_paragraph(p, size=9.5)

    _add_section_heading(doc, "Do This Now Checklist and Implementation Roadmap")
    roadmap = doc.add_table(rows=4, cols=3)
    roadmap.alignment = WD_TABLE_ALIGNMENT.CENTER
    header = ["Timeline", "Action Items", "Purpose"]
    rows = [
        ["Next 30 Days", "Open Solo 401(k) or SEP IRA; confirm 1099 records; organize Schedule C documentation; set up separate tax savings account.", "Build the foundation without disrupting the current filing/reporting system."],
        ["Next 90 Days", "Review contractor classification; compare vehicle methods; create monthly profit dashboard; review children employment facts.", "Reduce compliance risk and identify high-impact strategies before year-end."],
        ["Next 12 Months", "Run S-Corp feasibility at each quarter-end; implement payroll if profit supports it; set annual retirement contribution target; schedule quarterly tax planning reviews.", "Convert the business from reactive tax prep to proactive tax planning."],
    ]
    for i, val in enumerate(header):
        roadmap.cell(0, i).text = val
        _format_cell(roadmap.cell(0, i), bold=True, color="FFFFFF", fill=BRAND_RED)
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, val in enumerate(row):
            roadmap.cell(r_idx, c_idx).text = val
            _format_cell(roadmap.cell(r_idx, c_idx), fill="FFFFFF")

    _add_callout(doc, "Do This Now - Advisor Directive", "Start with documentation, contractor compliance, retirement plan setup, and profit tracking. Do not rush into an S-Corp until the profit level supports it. The strongest recommendation is to build the structure now so the client is ready when profit increases.", fill="FFF7DD")

    _add_section_heading(doc, "Final Advisor Recommendation")
    p = doc.add_paragraph("This client has a strong business revenue base and a favorable family-credit profile, but the current tax picture is not yet optimized. The correct planning path is to protect existing deductions, improve documentation, increase net profit intentionally, and then use entity structure, retirement funding, child employment, and vehicle strategy to control taxes. The highest-value planning message is simple: do not stay small to avoid tax. Build profit, then control tax through structure.")
    _style_paragraph(p, size=10)

    doc.save(output_path)
    return output_path


def demo_generate_valhalla_docx():
    sample = {
        "client_name": "Nathan Deratany",
        "tax_year": 2025,
        "filing_status": "Head of Household",
        "dependents": 2,
        "agi": 24266,
        "taxable_income": 0,
        "total_tax": 3429,
        "refund": 6731,
        "schedule_c_gross_revenue": 216265,
        "schedule_c_net_profit": 24266,
        "logo_path": "valhalla_logo.jpg",
    }
    return generate_valhalla_docx_report(sample)
