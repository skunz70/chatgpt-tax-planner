from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def money(value):
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"


def add_red_header(paragraph, text):
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(150, 0, 0)


def generate_valhalla_docx_report(data: dict, output_path="valhalla_premium_report.docx"):
    from valhalla_strategy_engine import generate_dynamic_tax_strategy

    doc = Document()

    client_name = data.get("client_name", "Client")
    tax_year = data.get("tax_year", "2025")
    filing_status = data.get("filing_status", "MFJ")
    agi = data.get("agi", 0)
    taxable_income = data.get("taxable_income", 0)
    total_tax = data.get("total_tax", 0)
    federal_withholding = data.get("federal_withholding", 0)
    schedule_c_gross = data.get("schedule_c_gross_revenue", 0)
    schedule_c_profit = data.get("schedule_c_net_profit", 0)
    contract_labor = data.get("contract_labor", 0)
    capital_gains = data.get("capital_gains", 0)
    state = data.get("state", "AZ")

    strategy_result = generate_dynamic_tax_strategy(data)
    priority_actions = strategy_result.get("priority_actions", [])
    dynamic_sections = strategy_result.get("dynamic_sections", [])

    projected_balance = float(total_tax or 0) - float(federal_withholding or 0)

    # ===== TITLE =====
    title = doc.add_paragraph()
    title_run = title.add_run("VALHALLA TAX SERVICES\nComprehensive Tax Strategy Report")
    title_run.bold = True
    title_run.font.size = Pt(18)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT

    doc.add_paragraph(f"Client: {client_name}")
    doc.add_paragraph(f"Tax Year: {tax_year}")
    doc.add_paragraph("Prepared by: Scott Kunz, ChFC, TPCP, Enrolled Agent and Financial Advisor")
    doc.add_paragraph("")

    # ===== EXECUTIVE SUMMARY =====
    p = doc.add_paragraph()
    add_red_header(p, "EXECUTIVE SUMMARY")

    doc.add_paragraph(
        f"This report analyzes the supplied tax facts for {client_name}. "
        f"The client is filing {filing_status}, has AGI of {money(agi)}, taxable income of {money(taxable_income)}, "
        f"and total federal tax of {money(total_tax)}. The planning focus is to identify strategies supported by the actual client data, "
        f"not generic tax-planning ideas."
    )

    # ===== CONFIRMED TAX POSITION =====
    p = doc.add_paragraph()
    add_red_header(p, "CONFIRMED TAX POSITION")

    table = doc.add_table(rows=2, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    headers = ["Filing Status", "AGI", "Taxable Income", "Total Tax", "Fed Withholding"]
    values = [filing_status, money(agi), money(taxable_income), money(total_tax), money(federal_withholding)]

    for i, h in enumerate(headers):
        table.cell(0, i).text = h

    for i, v in enumerate(values):
        table.cell(1, i).text = str(v)

    # ===== WITHHOLDING POSITION =====
    p = doc.add_paragraph()
    add_red_header(p, "WITHHOLDING / PAYMENT POSITION")

    if projected_balance > 0:
        doc.add_paragraph(
            f"Based on supplied data, total federal tax exceeds federal withholding by approximately {money(projected_balance)}. "
            f"This indicates a projected balance due unless additional payments, credits, or withholding exist outside the supplied facts."
        )
    else:
        doc.add_paragraph(
            f"Based on supplied data, federal withholding exceeds total federal tax by approximately {money(abs(projected_balance))}. "
            f"This indicates a projected overpayment before considering other payments, credits, or adjustments."
        )

    # ===== SCHEDULE C =====
    if float(schedule_c_gross or 0) > 0 or float(schedule_c_profit or 0) > 0:
        p = doc.add_paragraph()
        add_red_header(p, "SCHEDULE C BUSINESS ANALYSIS")

        business_table = doc.add_table(rows=2, cols=4)
        business_table.alignment = WD_TABLE_ALIGNMENT.CENTER

        headers = ["Gross Revenue", "Net Profit", "Contract Labor", "State"]
        values = [money(schedule_c_gross), money(schedule_c_profit), money(contract_labor), state]

        for i, h in enumerate(headers):
            business_table.cell(0, i).text = h

        for i, v in enumerate(values):
            business_table.cell(1, i).text = str(v)

        doc.add_paragraph(
            f"The Schedule C activity shows gross revenue of {money(schedule_c_gross)} and net profit of {money(schedule_c_profit)}. "
            f"Planning should focus on substantiation, contractor classification, retirement plan design, QBI support, and entity timing."
        )

    # ===== CAPITAL GAINS =====
    if float(capital_gains or 0) > 0:
        p = doc.add_paragraph()
        add_red_header(p, "CAPITAL GAIN PLANNING")

        doc.add_paragraph(
            f"The supplied facts include capital gains of {money(capital_gains)}. "
            f"Future gain harvesting, loss harvesting, and bracket management should be coordinated with ordinary income and taxable income levels."
        )

    # ===== PRIORITY ACTIONS =====
    p = doc.add_paragraph()
    add_red_header(p, "TOP PRIORITY ACTIONS")

    if priority_actions:
        table = doc.add_table(rows=1 + len(priority_actions), cols=4)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        table.cell(0, 0).text = "Priority"
        table.cell(0, 1).text = "Recommendation"
        table.cell(0, 2).text = "Estimated Impact"
        table.cell(0, 3).text = "Timing"

        for i, action in enumerate(priority_actions, start=1):
            table.cell(i, 0).text = action.get("priority", "")
            table.cell(i, 1).text = action.get("title", "")
            table.cell(i, 2).text = str(action.get("estimated_savings", ""))
            table.cell(i, 3).text = action.get("timeline", "")

            doc.add_paragraph(
                f"{i}. {action.get('title', '')}: {action.get('reason', '')}"
            )
    else:
        doc.add_paragraph("No priority actions were generated from the supplied facts.")

    # ===== DYNAMIC STRATEGY SECTIONS =====
    if dynamic_sections:
        p = doc.add_paragraph()
        add_red_header(p, "CLIENT-SPECIFIC STRATEGY MODULES")

        for section in dynamic_sections:
            doc.add_paragraph(section.get("section", "Strategy"))
            doc.add_paragraph(section.get("body", ""))

    # ===== DO THIS NOW =====
    p = doc.add_paragraph()
    add_red_header(p, "DO THIS NOW CHECKLIST")

    if priority_actions:
        for action in priority_actions[:5]:
            doc.add_paragraph(f"- {action.get('title', '')}: {action.get('timeline', '')}")
    else:
        doc.add_paragraph("- Review client data and rerun report with complete inputs.")

    # ===== FINAL RECOMMENDATION =====
    p = doc.add_paragraph()
    add_red_header(p, "FINAL ADVISOR RECOMMENDATION")

    doc.add_paragraph(
        "The recommended next step is to begin with the highest-ranked planning items above, confirm all supporting documentation, "
        "and model the actual tax impact before implementation. This report is intended to convert the supplied tax data into a practical advisory roadmap."
    )

    doc.save(output_path)
    return output_path