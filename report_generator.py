# report_generator.py

from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile
import os
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

def safe_text(value):
    if value is None:
        return ""
    text = str(value)
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "•": "-",
        "→": "->",
        "✅": "",
        "⚠️": "",
        "❌": "",
        "·": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.encode("latin-1", "ignore").decode("latin-1")
class TaxReportPDF(FPDF):
    def header(self):
        if hasattr(self, 'logo_path') and self.logo_path:
            self.image(self.logo_path, x=245, y=5, w=45)  # Adjust right-aligned logo
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, self.title, ln=True, align="L")

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100)
        self.cell(
            0,
            10,
            "Valhalla Tax Services · 7055 W Bell Rd, Suite B20, Glendale, AZ 85308 · "
            "(623) 887-7921 · skunz@valhallataxservice.com · www.valhallataxservice.com",
            align="C",
        )

    def add_section(self, title, content):
        self.set_font("Helvetica", "B", 12)
        self.ln(8)
        self.cell(0, 10, safe_text(title), ln=True)
        self.set_font("Helvetica", "", 11)
        for line in str(content).split("\n"):
            self.multi_cell(0, 8, safe_text(line))

    def add_chart_image(self, img_path):
        self.ln(10)
        self.image(img_path, x=30, w=220)
        self.ln(5)

def generate_bar_chart(title, labels, values, filename):
    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("Scenario")
    plt.ylabel("Tax Liability")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_tax_plan_pdf(data, logo_path=None):
    pdf = TaxReportPDF(orientation="L", unit="mm", format="A4")
    pdf.logo_path = logo_path
    pdf.title = "Tax Planning Report"
    pdf.add_page()

    # Add summary section
    summary = f"""
    Filing Status: {data['filing_status']}
    AGI: ${data['agi']:,.2f}
    Taxable Income: ${data['taxable_income']:,.2f}
    Total Tax: ${data['total_tax']:,.2f}
    Marginal Rate: {data.get('marginal_rate', 'N/A')}
    """
    pdf.add_section("Federal Tax Summary", summary)

    # Optional: Add Strategy Recommendations
    if "strategies" in data:
        strategy_text = "\n".join(f"- {safe_text(s)}" for s in data["strategies"])
        pdf.add_section("Recommended Strategies", strategy_text)

    # Optional: Add Chart
    if "comparison_chart_data" in data:
        labels = data["comparison_chart_data"]["labels"]
        values = data["comparison_chart_data"]["values"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
            generate_bar_chart("Tax Scenario Comparison", labels, values, tmp_chart.name)
            pdf.add_chart_image(tmp_chart.name)
            os.unlink(tmp_chart.name)

    # Output as binary
    return pdf.output(dest="S").encode("latin1")
from fpdf import FPDF

def generate_smart_strategy_pdf(payload: dict) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="Letter")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Logo
    try:
        pdf.image("Valhalla Logo Eagle-Tax Services.jpg", x=160, y=10, w=40)
    except:
        pass  # Fails silently if logo missing

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 20, "Smart Strategy Report", ln=True)

    # Section: Summary
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Filing Status: {payload['filing_status'].title()}", ln=True)
    pdf.cell(0, 10, f"Adjusted Gross Income (AGI): ${payload['agi']:,.2f}", ln=True)
    pdf.cell(0, 10, f"Taxable Income: ${payload['taxable_income']:,.2f}", ln=True)
    pdf.cell(0, 10, f"Estimated Tax: ${payload['estimated_tax']:,.2f}", ln=True)

    # Section: Strategy Summary
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Suggested Strategies", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for strategy in payload["strategy_summary"]:
        pdf.multi_cell(0, 8, safe_text(f"- {strategy}"))

    # Section: Phaseout Thresholds
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Phaseout/Threshold Alerts", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for flag in payload["threshold_flags"]:
        pdf.multi_cell(0, 8, safe_text(f"- {flag}"))

    # Footer
    pdf.set_y(-30)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 10, "Valhalla Tax Services | www.valhallataxservice.com | (623) 887-7921", ln=True, align="C")

    return pdf.output(dest="S").encode("latin1")


def generate_valhalla_premium_tax_report(data: dict, output_path: str = "valhalla_premium_report.pdf"):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ValhallaTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        textColor=colors.HexColor("#0F1C2E"),
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "ValhallaSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.HexColor("#1F4E78"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ValhallaBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1A1A1A"),
    )

    default_values = {
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
        "contract_labor": 107920,
        "depreciation": 16624,
        "vehicle_deduction": 4628,
        "qbi_deduction": 4510,
    }
    payload = {**default_values, **(data or {})}

    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(letter),
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    tax_eff_rate = (payload["total_tax"] / payload["agi"] * 100) if payload["agi"] else 0
    strategy_savings = payload["depreciation"] + payload["vehicle_deduction"] + payload["qbi_deduction"]

    story = [
        Paragraph("Valhalla Premium Tax Strategy Report", title_style),
        Paragraph(
            f"Client: <b>{safe_text(payload['client_name'])}</b> | Tax Year: <b>{payload['tax_year']}</b>",
            body_style,
        ),
        Spacer(1, 8),
    ]

    def section(title, bullet_points):
        story.append(Paragraph(title, section_style))
        for point in bullet_points:
            story.append(Paragraph(f"• {safe_text(point)}", body_style))
        story.append(Spacer(1, 6))

    section("Executive Summary", [
        f"Current filing status is {payload['filing_status']} with {payload['dependents']} dependents.",
        f"AGI of ${payload['agi']:,.0f} and total tax of ${payload['total_tax']:,.0f} produced a refund of ${payload['refund']:,.0f}.",
        "Primary opportunities are concentrated in business deductions, QBI optimization, and cash-flow timing.",
    ])

    table_data = [
        ["Confirmed Tax Data", "Amount"],
        ["Adjusted Gross Income", f"${payload['agi']:,.0f}"],
        ["Taxable Income", f"${payload['taxable_income']:,.0f}"],
        ["Total Tax", f"${payload['total_tax']:,.0f}"],
        ["Refund", f"${payload['refund']:,.0f}"],
    ]
    table = Table(table_data, colWidths=[3.4 * inch, 2.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CFD8E3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FB")]),
    ]))
    story.extend([Paragraph("Confirmed Tax Data", section_style), table, Spacer(1, 8)])

    section("Federal Tax Analysis", [
        f"Effective federal tax rate is approximately {tax_eff_rate:.1f}% based on AGI.",
        "QBI deduction is currently captured and should be preserved through income smoothing.",
        "Evaluate retirement contribution layering to protect future year tax brackets.",
    ])

    section("Arizona Tax Analysis", [
        "Review conformity impacts between federal deductions and Arizona treatment.",
        "Maintain complete support for business-use expenses to reduce audit exposure.",
    ])

    section("Schedule C Business Analysis", [
        f"Gross revenue: ${payload['schedule_c_gross_revenue']:,.0f}; net profit: ${payload['schedule_c_net_profit']:,.0f}.",
        f"Contract labor spend of ${payload['contract_labor']:,.0f} is material and should be documented by vendor.",
        f"Depreciation (${payload['depreciation']:,.0f}) and vehicle deduction (${payload['vehicle_deduction']:,.0f}) are key levers.",
    ])

    section("Strategy Savings Summary", [
        f"Tracked deduction value from depreciation, vehicle, and QBI totals ${strategy_savings:,.0f}.",
        "Apply quarterly review cadence to convert year-end surprises into planned savings.",
    ])

    section("Top 3 Priority Actions", [
        "Finalize accountable plan and expense substantiation package.",
        "Implement monthly bookkeeping close to improve deduction capture.",
        "Schedule mid-year projection to tune withholding and estimated payments.",
    ])

    section("Do This Now Checklist", [
        "Collect receipts and mileage logs for all business-use vehicle activity.",
        "Reconcile contractor payments to 1099 records and W-9 files.",
        "Set calendar reminders for quarterly strategy reviews.",
    ])

    section("Multi-Year Strategy Roadmap", [
        "Year 1: tighten documentation and stabilize baseline taxable income.",
        "Year 2: expand retirement and entity-structure efficiency planning.",
        "Year 3: optimize long-term depreciation, exit planning, and family tax integration.",
    ])

    section("Advisor Summary / Signature Block", [
        "Prepared by: Valhalla Tax Strategy Team",
        "Advisor Signature: ______________________    Date: ______________________",
    ])

    doc.build(story)
    return output_path


def demo_generate_valhalla_premium_tax_report(output_path: str = "valhalla_premium_report.pdf"):
    return generate_valhalla_premium_tax_report(data={}, output_path=output_path)
