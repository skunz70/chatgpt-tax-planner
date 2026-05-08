from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def _num(value, default=0):
    try:
        if value is None or value == "":
            return float(default)
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "").strip()
        return float(value)
    except Exception:
        return float(default)


def money(value):
    return f"${_num(value):,.0f}"


def add_red_header(paragraph, text):
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(150, 0, 0)


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for col, header in enumerate(headers):
        table.cell(0, col).text = str(header)

    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row):
            table.cell(r, c).text = str(value)

    return table


def generate_valhalla_docx_report(data: dict, output_path="valhalla_premium_report.docx"):
    from valhalla_strategy_engine import generate_dynamic_tax_strategy
    from valhalla_roi_engine import generate_roi_analysis

    data = data or {}

    doc = Document()

    client_name = data.get("client_name", "Client")
    tax_year = data.get("tax_year", "2025")
    filing_status = data.get("filing_status", "MFJ")
    state = data.get("state", "AZ")

    agi = _num(data.get("agi", 0))
    taxable_income = _num(data.get("taxable_income", 0))
    total_tax = _num(data.get("total_tax", 0))
    federal_withholding = _num(data.get("federal_withholding", 0))

    schedule_c_gross = _num(data.get("schedule_c_gross_revenue", 0))
    schedule_c_profit = _num(data.get("schedule_c_net_profit", 0))
    contract_labor = _num(data.get("contract_labor", 0))
    capital_gains = _num(data.get("capital_gains", 0))
    age = _num(data.get("age", 0))

    projected_balance = total_tax - federal_withholding

    strategy_result = generate_dynamic_tax_strategy(data)
    roi_result = generate_roi_analysis(data)

    priority_actions = strategy_result.get("priority_actions", [])
    dynamic_sections = strategy_result.get("dynamic_sections", [])
    roi_items = roi_result.get("roi_items", [])

    title = doc.add_paragraph()
    title_run = title.add_run("VALHALLA TAX SERVICES\nComprehensive Tax Strategy Report")
    title_run.bold = True
    title_run.font.size = Pt(18)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT

    doc.add_paragraph(f"Client: {client_name}")
    doc.add_paragraph(f"Tax Year: {tax_year}")
    doc.add_paragraph("Prepared by: Scott Kunz, ChFC, TPCP, Enrolled Agent and Financial Advisor")
    doc.add_paragraph("")

    p = doc.add_paragraph()
    add_red_header(p, "EXECUTIVE SUMMARY")

    doc.add_paragraph(
        f"This report converts supplied tax facts into an advisor-level planning roadmap for {client_name}. "
        f"The client is filing {filing_status}, has AGI of {money(agi)}, taxable income of {money(taxable_income)}, "
        f"and total federal tax of {money(total_tax)}. Recommendations below are generated from the supplied data, "
        f"including business profit, withholding, capital gains, retirement facts, age, and state."
    )

    p = doc.add_paragraph()
    add_red_header(p, "CONFIRMED TAX POSITION")

    add_table(
        doc,
        ["Filing Status", "State", "AGI", "Taxable Income", "Total Tax", "Federal Withholding"],
        [[filing_status, state, money(agi), money(taxable_income), money(total_tax), money(federal_withholding)]],
    )

    p = doc.add_paragraph()
    add_red_header(p, "WITHHOLDING / PAYMENT POSITION")

    if projected_balance > 0:
        doc.add_paragraph(
            f"Based only on supplied federal tax and withholding data, the client is underpaid by approximately "
            f"{money(projected_balance)}. This should be reviewed immediately for withholding changes, estimated payments, "
            f"or confirmation of additional payments not included in the supplied data."
        )
    else:
        doc.add_paragraph(
            f"Based only on supplied federal tax and withholding data, the client appears overpaid by approximately "
            f"{money(abs(projected_balance))}. Confirm whether other payments, credits, or refund offsets apply."
        )

    if schedule_c_gross > 0 or schedule_c_profit > 0:
        p = doc.add_paragraph()
        add_red_header(p, "SCHEDULE C BUSINESS ANALYSIS")

        expense_estimate = max(0, schedule_c_gross - schedule_c_profit)
        profit_margin = (schedule_c_profit / schedule_c_gross * 100) if schedule_c_gross else 0

        add_table(
            doc,
            ["Gross Revenue", "Estimated Expenses", "Net Profit", "Profit Margin", "Contract Labor"],
            [[money(schedule_c_gross), money(expense_estimate), money(schedule_c_profit), f"{profit_margin:.1f}%", money(contract_labor)]],
        )

        doc.add_paragraph(
            f"The Schedule C activity reports gross revenue of {money(schedule_c_gross)} and net profit of "
            f"{money(schedule_c_profit)}, producing an estimated profit margin of {profit_margin:.1f}%. "
            f"Planning should focus on substantiation, contractor classification, retirement plan design, QBI support, "
            f"estimated tax planning, and entity timing."
        )

    if capital_gains > 0:
        p = doc.add_paragraph()
        add_red_header(p, "CAPITAL GAIN PLANNING")

        doc.add_paragraph(
            f"The supplied facts include capital gains of {money(capital_gains)}. Future planning should use the "
            f"0% / 15% / 20% long-term capital gain framework and coordinate gain recognition with ordinary income, "
            f"loss harvesting, and portfolio rebalancing."
        )

    p = doc.add_paragraph()
    add_red_header(p, "TOP PRIORITY ACTIONS")

    if priority_actions:
        rows = []
        for item in priority_actions:
            rows.append([
                item.get("priority", ""),
                item.get("title", ""),
                item.get("estimated_savings", ""),
                item.get("timeline", ""),
            ])

        add_table(
            doc,
            ["Priority", "Recommendation", "Estimated Impact", "Timing"],
            rows,
        )

        for index, item in enumerate(priority_actions, start=1):
            doc.add_paragraph(
                f"{index}. {item.get('title', '')}: {item.get('reason', '')}"
            )
    else:
        doc.add_paragraph("No priority strategies were generated from the supplied facts.")

    p = doc.add_paragraph()
    add_red_header(p, "ROI-RANKED STRATEGY SCORECARD")

    if roi_items:
        rows = []
        for item in roi_items:
            rows.append([
                item.get("strategy", ""),
                item.get("estimated_value", ""),
                item.get("score", ""),
                item.get("difficulty", ""),
                item.get("timeline", ""),
            ])

        add_table(
            doc,
            ["Strategy", "Estimated Value", "Score", "Difficulty", "Timeline"],
            rows,
        )
    else:
        doc.add_paragraph("No ROI items were generated from the supplied facts.")

    if dynamic_sections:
        p = doc.add_paragraph()
        add_red_header(p, "CLIENT-SPECIFIC STRATEGY MODULES")

        for section in dynamic_sections:
            heading = doc.add_paragraph()
            heading_run = heading.add_run(section.get("section", "Strategy"))
            heading_run.bold = True
            heading_run.font.size = Pt(11)

            doc.add_paragraph(section.get("body", ""))

    p = doc.add_paragraph()
    add_red_header(p, "DO THIS NOW CHECKLIST AND IMPLEMENTATION ROADMAP")

    if priority_actions:
        for item in priority_actions[:5]:
            doc.add_paragraph(
                f"- {item.get('title', '')}: {item.get('timeline', '')}. {item.get('reason', '')}"
            )
    else:
        doc.add_paragraph("- Gather complete client tax facts and rerun the premium planning report.")

    p = doc.add_paragraph()
    add_red_header(p, "FINAL ADVISOR RECOMMENDATION")

    if priority_actions:
        top = priority_actions[0]
        doc.add_paragraph(
            f"The first action item should be: {top.get('title', '')}. "
            f"This ranks highest based on the supplied facts and estimated planning impact. "
            f"Before implementation, confirm supporting documentation, final tax data, state impact, and cash-flow timing."
        )
    else:
        doc.add_paragraph(
            "The recommended next step is to confirm complete client tax data and rerun the strategy model."
        )

    doc.add_paragraph("")
    doc.add_paragraph("PREPARED BY")
    doc.add_paragraph("Scott Kunz, ChFC, TPCP")
    doc.add_paragraph("Enrolled Agent and Financial Advisor")
    doc.add_paragraph("Valhalla Tax Services")
    doc.add_paragraph("7055 W Bell Rd, Suite B20, Glendale, AZ 85308")
    doc.add_paragraph("(623) 887-7921")
    doc.add_paragraph("skunz@valhallataxservice.com")
    doc.add_paragraph("www.valhallataxservice.com")

    doc.save(output_path)
    return output_path