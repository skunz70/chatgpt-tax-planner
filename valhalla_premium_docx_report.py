from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH


def generate_valhalla_docx_report(data: dict, output_path: str = "valhalla_premium_report.docx"):
    doc = Document()

    # Title
    title = doc.add_paragraph()
    run = title.add_run("VALHALLA TAX SERVICES\nComprehensive Tax Strategy Report")
    run.bold = True
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT

    doc.add_paragraph(f"Client: {data.get('client_name','Client')}")
    doc.add_paragraph(f"Tax Year: {data.get('tax_year','2025')}")

    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph("This return reflects a Schedule C-driven client with primary exposure to self-employment tax. Planning should focus on structure, not just deductions.")

    doc.add_heading("Confirmed Tax Data", level=1)
    table = doc.add_table(rows=4, cols=2)
    table.cell(0,0).text = "AGI"
    table.cell(0,1).text = str(data.get("agi",0))
    table.cell(1,0).text = "Taxable Income"
    table.cell(1,1).text = str(data.get("taxable_income",0))
    table.cell(2,0).text = "Total Tax"
    table.cell(2,1).text = str(data.get("total_tax",0))
    table.cell(3,0).text = "Refund"
    table.cell(3,1).text = str(data.get("refund",0))

    doc.add_heading("Top 3 Priority Actions", level=1)
    doc.add_paragraph("1. Clean up Schedule C structure")
    doc.add_paragraph("2. Implement retirement strategy")
    doc.add_paragraph("3. Prepare for S-Corp timing")

    doc.save(output_path)
    return output_path


def demo_generate_valhalla_docx():
    sample = {
        "client_name": "Nathan Deratany",
        "tax_year": 2025,
        "agi": 24266,
        "taxable_income": 0,
        "total_tax": 3429,
        "refund": 6731
    }
    return generate_valhalla_docx_report(sample)
