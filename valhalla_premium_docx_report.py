import os
import tempfile
from typing import Any, Dict, List

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


VALHALLA_RED = "9D1B24"
LIGHT_RED = "F3E6E8"
LIGHT_GOLD = "F7EFD6"
LIGHT_GRAY = "F2F2F2"


def _num(value: Any, default: float = 0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "").replace("%", "").strip()
        return float(value)
    except Exception:
        return float(default)


def money(value: Any) -> str:
    return f"${_num(value):,.0f}"


def pct(value: Any) -> str:
    return f"{_num(value):.1f}%"


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text, bold=False, color=None, size=9):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing = 1.08
    run = p.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cell_margins(cell, top=95, start=110, bottom=95, end=110):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.find(qn("w:tcMar"))
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin_key, margin_value in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        margin = tc_mar.find(qn(f"w:{margin_key}"))
        if margin is None:
            margin = OxmlElement(f"w:{margin_key}")
            tc_mar.append(margin)
        margin.set(qn("w:w"), str(margin_value))
        margin.set(qn("w:type"), "dxa")


def set_table_column_widths(table, widths):
    table.autofit = False
    for row in table.rows:
        for idx, width in enumerate(widths):
            if idx < len(row.cells):
                row.cells[idx].width = width
                set_cell_margins(row.cells[idx])


def add_header_footer(section):
    header = section.header
    p = header.paragraphs[0]
    p.text = "VALHALLA TAX SERVICES | Comprehensive Tax Strategy Report"
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in p.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(90, 90, 90)

    footer = section.footer
    p = footer.paragraphs[0]
    p.text = "Prepared by Scott Kunz, ChFC, TPCP, Enrolled Agent and Financial Advisor | Confidential client planning document"
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in p.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(120, 120, 120)


def add_section_title(doc, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(title)
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor.from_string(VALHALLA_RED)

    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), VALHALLA_RED)
    border.append(bottom)
    p._p.get_or_add_pPr().append(border)


def add_box(doc, title, body, fill=LIGHT_RED):
    table = doc.add_table(rows=2, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_column_widths(table, [Inches(9.9)])

    set_cell_shading(table.cell(0, 0), VALHALLA_RED)
    set_cell_text(table.cell(0, 0), title, bold=True, color="FFFFFF", size=9)

    set_cell_shading(table.cell(1, 0), fill)
    set_cell_text(table.cell(1, 0), body, size=8)

    spacer = doc.add_paragraph("")
    spacer.paragraph_format.space_after = Pt(5)


def add_table(doc, headers: List[str], rows: List[List[Any]], header_fill=VALHALLA_RED, col_widths=None, compact=False):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    if col_widths:
        set_table_column_widths(table, col_widths)

    for i, h in enumerate(headers):
        set_cell_shading(table.cell(0, i), header_fill)
        set_cell_text(table.cell(0, i), h, bold=True, color="FFFFFF", size=8)
    set_repeat_table_header(table.rows[0])
    table.rows[0].height = Pt(18)

    for r, row in enumerate(rows, start=1):
        table.rows[r].height = Pt(20 if not compact else 18)
        for c, value in enumerate(row):
            set_cell_text(table.cell(r, c), value, size=7 if compact else 8)

    spacer = doc.add_paragraph("")
    spacer.paragraph_format.space_after = Pt(6)
    return table


def add_bullets(doc, items: List[str]):
    for item in items:
        p = doc.add_paragraph(style=None)
        p.paragraph_format.left_indent = Inches(0.15)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(f"- {item}")
        run.font.size = Pt(9)


def make_bar_chart(labels, values, title, ylabel, filename):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    bars = ax.bar(labels, values)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.bar_label(bars, fmt="%.0f", fontsize=7)
    fig.tight_layout()

    path = os.path.join(tempfile.gettempdir(), filename)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def safe_strategy_output(data: Dict) -> Dict:
    try:
        from valhalla_strategy_engine import generate_dynamic_tax_strategy
        result = generate_dynamic_tax_strategy(data)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return {"priority_actions": [], "dynamic_sections": []}


def safe_roi_output(data: Dict) -> Dict:
    try:
        from valhalla_roi_engine import generate_roi_analysis
        result = generate_roi_analysis(data)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return {"roi_items": []}


def normalize_priority_actions(data: Dict, priority_actions: List[Dict]) -> List[Dict]:
    if priority_actions:
        return priority_actions[:3]

    schedule_c_profit = _num(data.get("schedule_c_net_profit", 0))
    contract_labor = _num(data.get("contract_labor", 0))
    has_retirement = bool(data.get("has_retirement_accounts", False))

    fallback = []

    if contract_labor > 0:
        fallback.append({
            "title": "Clean up Schedule C structure and contractor compliance",
            "estimated_savings": f"Risk reduction plus protects {money(contract_labor)}+ deductions",
            "timeline": "Now to 90 days",
            "reason": "This protects the largest deduction category and reduces audit exposure.",
        })

    if has_retirement or schedule_c_profit > 0:
        fallback.append({
            "title": "Review retirement funding and Roth coordination",
            "estimated_savings": "Bracket-management and long-term tax control",
            "timeline": "Current year",
            "reason": "Creates a repeatable tax and wealth-building strategy.",
        })

    fallback.append({
        "title": "Run annual withholding and Arizona cash-flow review",
        "estimated_savings": "Cash-flow and penalty protection",
        "timeline": "Immediate",
        "reason": "Confirms whether current withholding aligns with projected federal and state tax.",
    })

    return fallback[:3]


def generate_valhalla_docx_report(data: dict, output_path="valhalla_premium_report.docx"):
    data = data or {}

    client_name = data.get("client_name", "Client")
    tax_year = data.get("tax_year", "2025")
    filing_status = data.get("filing_status", "Unknown")
    state = data.get("state", "AZ")

    agi = _num(data.get("agi", 0))
    taxable_income = _num(data.get("taxable_income", 0))
    total_tax = _num(data.get("total_tax", 0))
    federal_withholding = _num(data.get("federal_withholding", 0))
    refund = _num(data.get("refund", 0))

    schedule_c_gross = _num(data.get("schedule_c_gross_revenue", 0))
    schedule_c_profit = _num(data.get("schedule_c_net_profit", 0))
    contract_labor = _num(data.get("contract_labor", 0))
    depreciation = _num(data.get("depreciation", data.get("depreciation_expense", 0)))
    vehicle_expense = _num(data.get("vehicle_expense", data.get("car_truck_expense", 0)))
    capital_gains = _num(data.get("capital_gains", 0))
    age = _num(data.get("age", 0))

    estimated_expenses = max(0, schedule_c_gross - schedule_c_profit)
    profit_margin = (schedule_c_profit / schedule_c_gross * 100) if schedule_c_gross else 0
    se_tax_estimate = round(schedule_c_profit * 0.9235 * 0.153, 0) if schedule_c_profit else 0
    income_tax_estimate = max(0, total_tax - se_tax_estimate)
    withholding_delta = federal_withholding - total_tax

    strategy_result = safe_strategy_output(data)
    roi_result = safe_roi_output(data)
    priority_actions = normalize_priority_actions(data, strategy_result.get("priority_actions", []))
    dynamic_sections = strategy_result.get("dynamic_sections", [])
    roi_items = roi_result.get("roi_items", [])

    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.top_margin = Inches(0.35)
    section.bottom_margin = Inches(0.35)
    section.left_margin = Inches(0.45)
    section.right_margin = Inches(0.45)
    add_header_footer(section)

    styles = doc.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(9)

    # PAGE 1
    branding = doc.add_table(rows=1, cols=2)
    branding.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_column_widths(branding, [Inches(2.2), Inches(7.7)])

    logo_cell = branding.cell(0, 0)
    logo_cell.text = ""
    logo_paths = ["valhalla_logo.png", "valhalla_logo.jpg", "Valhalla Logo Eagle-Tax Services.jpg"]
    for logo_path in logo_paths:
        if os.path.exists(logo_path):
            logo_cell.paragraphs[0].add_run().add_picture(logo_path, width=Inches(1.85))
            break

    title_cell = branding.cell(0, 1)
    title_cell.text = ""
    title = title_cell.paragraphs[0]
    title.paragraph_format.space_after = Pt(2)
    run = title.add_run("VALHALLA TAX SERVICES")
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = RGBColor.from_string(VALHALLA_RED)

    subtitle = title_cell.add_paragraph("Comprehensive Tax Strategy Report")
    subtitle.paragraph_format.space_after = Pt(4)
    subtitle_run = subtitle.runs[0]
    subtitle_run.bold = True
    subtitle_run.font.size = Pt(14)
    subtitle_run.font.color.rgb = RGBColor(60, 60, 60)

    meta = title_cell.add_paragraph()
    meta.paragraph_format.space_after = Pt(8)
    meta.add_run(f"Client: {client_name}   ").bold = True
    meta.add_run(f"Tax Year: {tax_year}   ").bold = True
    meta.add_run("Prepared by: Scott Kunz, ChFC, TPCP, Enrolled Agent")

    add_box(
        doc,
        "Advisor Summary",
        (
            "This engagement translates today’s return data into a forward-looking decision framework. We prioritize sequencing, "
            "cash-flow control, threshold triggers, and compliance durability so implementation occurs in the right order."
        ),
        LIGHT_RED,
    )

    add_section_title(doc, "CONFIRMED TAX POSITION")
    add_table(
        doc,
        ["Filing Status", "AGI", "Taxable Income", "Total Tax", "Federal Withholding", "Refund / Overpayment"],
        [[filing_status, money(agi), money(taxable_income), money(total_tax), money(federal_withholding), money(refund if refund else max(0, withholding_delta))]],
        col_widths=[Inches(1.3), Inches(1.25), Inches(1.55), Inches(1.2), Inches(1.8), Inches(2.0)],
    )

    doc.add_paragraph(
        "Source reviewed: supplied tax return data and planning facts. Amounts should be verified against final filed copies before implementation."
    )

    add_section_title(doc, "EXECUTIVE SUMMARY")
    add_table(
        doc,
        ["Current Position", "Advisor Conclusion"],
        [[
            (
                f"AGI is {money(agi)} and taxable income is {money(taxable_income)}. "
                f"Total federal tax is {money(total_tax)} with federal withholding of {money(federal_withholding)}."
            ),
            (
                "The advisor view is to stage decisions by threshold: protect compliance first, then optimize entity and retirement "
                "structure once profit and liquidity levels justify complexity."
            ),
        ]],
        col_widths=[Inches(4.8), Inches(5.1)],
        compact=True,
    )

    if withholding_delta >= 0:
        primary_message = (
            f"The client is not currently showing an underpayment problem. Federal withholding exceeds federal tax by approximately "
            f"{money(withholding_delta)}. The highest-value next step is proactive bracket management: coordinate retirement "
            f"contributions, evaluate Roth capacity, and tune withholding for cash-flow efficiency while preserving 'not yet' options."
        )
    else:
        primary_message = (
            f"The client appears underpaid by approximately {money(abs(withholding_delta))}. The immediate planning target is payment "
            f"alignment: correct withholding and estimates now, then sequence structural strategies after cash-flow stabilization."
        )

    add_box(doc, "Primary Planning Message", primary_message, LIGHT_GOLD)

    doc.add_page_break()

    # PAGE 2
    add_section_title(doc, "CURRENT FEDERAL TAX ANALYSIS")

    chart1 = make_bar_chart(
        ["Federal\nIncome Tax", "Self-\nEmployment Tax"],
        [income_tax_estimate, se_tax_estimate],
        "Current Tax Burden: Income Tax vs SE Tax",
        "Current tax ($)",
        "valhalla_tax_burden.png",
    )

    analysis_table = doc.add_table(rows=1, cols=2)
    analysis_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_column_widths(analysis_table, [Inches(5.0), Inches(4.9)])
    left = analysis_table.cell(0, 0)
    right = analysis_table.cell(0, 1)

    left.text = ""
    left.paragraphs[0].add_run("Tax Burden Analysis").bold = True
    bullets = [
        f"Federal income tax estimate: {money(income_tax_estimate)} based on supplied total tax and SE tax estimate.",
        f"Self-employment tax estimate: {money(se_tax_estimate)} if Schedule C profit is present.",
        f"Federal withholding position: {'overpaid' if withholding_delta >= 0 else 'underpaid'} by approximately {money(abs(withholding_delta))}.",
        "This result should be reconciled with credits, estimated payments, and final return data before implementation.",
    ]
    for b in bullets:
        p = left.add_paragraph()
        p.add_run(f"- {b}").font.size = Pt(9)

    right.paragraphs[0].add_run().add_picture(chart1, width=Inches(4.4))

    if schedule_c_gross > 0 or schedule_c_profit > 0:
        add_section_title(doc, "SCHEDULE C BUSINESS ANALYSIS")
        add_table(
            doc,
            ["Business Metric", "Amount", "Advisor Read", "Planning Priority"],
            [
                ["Gross Revenue", money(schedule_c_gross), "Strong activity level" if schedule_c_gross > 100000 else "Developing activity level", "Set future-state structure and reporting cadence"],
                ["Total Expenses", f"~{money(estimated_expenses)}", "High expense ratio" if profit_margin < 20 else "Moderate expense ratio", "Strengthen substantiation controls before scaling deductions"],
                ["Net Profit", money(schedule_c_profit), f"About {profit_margin:.1f}% margin", "Use margin trend to trigger entity and retirement decisions"],
                ["Contract Labor", money(contract_labor), "Material compliance item" if contract_labor > 50000 else "Review if applicable", "Apply worker-classification decision framework before year-end filings"],
            ],
            col_widths=[Inches(1.5), Inches(1.4), Inches(2.6), Inches(4.4)],
            compact=True,
        )

    doc.add_page_break()

    # PAGE 3
    if schedule_c_gross > 0 or schedule_c_profit > 0:
        chart2 = make_bar_chart(
            ["Gross\nRevenue", "Expenses", "Net\nProfit"],
            [schedule_c_gross, estimated_expenses, schedule_c_profit],
            "Schedule C Revenue, Expenses, and Profit",
            "Dollars",
            "valhalla_schedule_c.png",
        )

        two_col = doc.add_table(rows=1, cols=2)
        two_col.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_column_widths(two_col, [Inches(4.9), Inches(5.0)])
        two_col.cell(0, 0).paragraphs[0].add_run().add_picture(chart2, width=Inches(4.6))

        right = two_col.cell(0, 1)
        right.text = ""
        r = right.paragraphs[0].add_run("Major Deduction Categories Reviewed")
        r.bold = True
        deduction_items = []
        if contract_labor:
            deduction_items.append(f"Contract labor: {money(contract_labor)} - large enough to require worker classification review and 1099 compliance support.")
        if depreciation:
            deduction_items.append(f"Depreciation and equipment: {money(depreciation)} - indicates asset investment and possible future Section 179 planning opportunities.")
        if vehicle_expense:
            deduction_items.append(f"Vehicle expense: {money(vehicle_expense)} - compare standard mileage vs actual expense planning.")
        deduction_items.append("Supplies, office, utilities, insurance, and other deductions should be supported by clean records.")

        for item in deduction_items:
            p = right.add_paragraph()
            p.add_run(f"- {item}").font.size = Pt(9)

        if contract_labor > 0:
            add_box(
                doc,
                "Schedule C Risk Point",
                (
                    "The contract labor amount is the largest compliance item. The planning recommendation is not to eliminate the deduction. "
                    "The recommendation is to document it properly, confirm independent contractor status, issue required Forms 1099, and evaluate "
                    "whether any workers should be moved to payroll as the business scales."
                ),
                LIGHT_GOLD,
            )

    add_section_title(doc, "TOP 3 PRIORITY ACTIONS")
    rows = []
    for idx, action in enumerate(priority_actions[:3], start=1):
        rows.append([
            idx,
            action.get("title", ""),
            action.get("estimated_savings", ""),
            action.get("timeline", ""),
            action.get("reason", ""),
        ])
    add_table(
        doc,
        ["Rank", "Recommendation", "Estimated Impact", "Timing", "Why It Matters"],
        rows,
        col_widths=[Inches(0.55), Inches(2.7), Inches(2.0), Inches(1.2), Inches(3.45)],
        compact=True,
    )

    add_section_title(doc, "STRATEGY IMPACT BY CATEGORY")

    doc.add_page_break()

    # PAGE 4
    impact_labels = []
    impact_values = []

    for item in roi_items[:5]:
        impact_labels.append(str(item.get("strategy", "Strategy"))[:14])
        impact_values.append(_num(str(item.get("estimated_value", "0")).replace("$", "").replace(",", "").split("-")[-1], 0))

    if not impact_labels:
        impact_labels = ["S-Corp\nFuture", "Retirement", "Contractor\nCompliance", "Capital\nGain", "Withholding"]
        impact_values = [
            7500 if schedule_c_profit >= 60000 else 0,
            max(0, schedule_c_profit * 0.20),
            10000 if contract_labor > 50000 else 0,
            capital_gains * 0.15,
            abs(withholding_delta),
        ]

    chart3 = make_bar_chart(
        impact_labels,
        impact_values,
        "Estimated Strategy Impact by Category",
        "Estimated tax impact ($)",
        "valhalla_strategy_impact.png",
    )

    two_col = doc.add_table(rows=1, cols=2)
    two_col.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_column_widths(two_col, [Inches(4.9), Inches(5.0)])
    two_col.cell(0, 0).paragraphs[0].add_run().add_picture(chart3, width=Inches(4.7))

    right = two_col.cell(0, 1)
    right.text = ""
    r = right.paragraphs[0].add_run("Estimated Planning Impact")
    r.bold = True
    impact_notes = [
        "S-Corp conversion is a threshold decision: proceed only when recurring profit can support reasonable compensation, payroll friction, and admin cost while still producing net savings.",
        "Retirement funding should be sequenced after quarterly profit visibility improves; contribution design should follow marginal bracket and liquidity targets, then Roth coordination.",
        "Contractor compliance is a deduction-protection strategy: documentation and classification discipline preserve deductions already claimed and reduce reclassification risk.",
        "Capital gain planning should use the 0% / 15% / 20% long-term capital gain framework.",
        "Withholding and estimated payments should be calibrated to a target outcome (small refund or small balance due) to improve monthly liquidity discipline.",
    ]
    for note in impact_notes:
        p = right.add_paragraph()
        p.add_run(f"- {note}").font.size = Pt(9)

    strategy_rows = []
    for action in priority_actions:
        strategy_rows.append([
            action.get("title", ""),
            "Applicable based on supplied facts",
            action.get("reason", ""),
            action.get("estimated_savings", ""),
        ])

    add_table(
        doc,
        ["Strategy", "Current-Year Applicability", "Modeled Scenario", "Estimated Tax Impact"],
        strategy_rows,
        col_widths=[Inches(2.6), Inches(2.1), Inches(3.9), Inches(1.3)],
        compact=True,
    )

    doc.add_page_break()

    # PAGE 5
    add_section_title(doc, "DETAILED STRATEGY NOTES")

    if dynamic_sections:
        for section_data in dynamic_sections:
            p = doc.add_paragraph()
            run = p.add_run(section_data.get("section", "Strategy"))
            run.bold = True
            run.font.size = Pt(11)
            doc.add_paragraph(section_data.get("body", ""))
    else:
        notes = [
            ("Retirement Plan Funding", "Retirement planning should be coordinated with taxable income, cash flow, and long-term bracket management."),
            ("Withholding Optimization", "Withholding should be adjusted only after confirming final payments, credits, refund targets, and Arizona tax impact."),
            ("Arizona Planning", "Arizona planning should be coordinated with federal taxable income, state withholding, estimated payments, and retirement strategy."),
        ]
        for title_text, body in notes:
            p = doc.add_paragraph()
            run = p.add_run(title_text)
            run.bold = True
            run.font.size = Pt(11)
            doc.add_paragraph(body)

    doc.add_page_break()

    # PAGE 6
    add_section_title(doc, "DO THIS NOW CHECKLIST AND IMPLEMENTATION ROADMAP")

    roadmap_rows = [
        [
            "Next 30 Days",
            "Confirm source data, withholding, retirement facts, Schedule C documentation, and Arizona payment position.",
            "Build the foundation before implementing advanced strategies.",
        ],
        [
            "Next 90 Days",
            "Model retirement funding, Roth conversion capacity, contractor compliance, capital gain timing, and cash-flow changes.",
            "Convert recommendations into measurable planning decisions.",
        ],
        [
            "Next 12 Months",
            "Review progress quarterly, update income projections, and adjust the strategy as income, deductions, and family facts change.",
            "Turn the report into a recurring advisory process.",
        ],
    ]

    add_table(
        doc,
        ["Timeline", "Action Items", "Purpose"],
        roadmap_rows,
        col_widths=[Inches(1.4), Inches(4.9), Inches(3.6)],
        compact=True,
    )

    add_box(
        doc,
        "Do This Now - Advisor Directive",
        (
            "Execute in sequence, not in parallel: validate data and documentation, stabilize withholding and cash flow, then implement "
            "threshold-qualified strategies with measurable checkpoints."
        ),
        LIGHT_GOLD,
    )

    add_section_title(doc, "FINAL ADVISOR RECOMMENDATION")
    doc.add_paragraph(
        "Advisor recommendation: use a staged implementation model with quarterly decision gates. Keep 'not yet' strategies on standby "
        "until profit, documentation, and liquidity thresholds are met; once triggered, execute quickly to capture current-year efficiency."
    )

    p = doc.add_paragraph()
    p.add_run("IMPORTANT PLANNING NOTES").bold = True
    doc.add_paragraph(
        "The savings estimates in this report are planning illustrations, not guaranteed outcomes. Actual savings depend on final income, filing status, business-use "
        "percentages, payroll requirements, documentation, state law, entity costs, and implementation timing. Strategies involving children, contractors, retirement plans, "
        "vehicle deductions, and S-Corp elections should be implemented with proper documentation and professional review."
    )

    doc.add_paragraph("Prepared by Scott Kunz, ChFC, TPCP, Enrolled Agent and Financial Advisor | Valhalla Tax Services")

    doc.save(output_path)
    return output_path
