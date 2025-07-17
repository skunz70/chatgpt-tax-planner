# report_generator.py

from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile
import os

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
        self.cell(0, 10, title, ln=True)
        self.set_font("Helvetica", "", 11)
        for line in content.split("\n"):
            self.multi_cell(0, 8, line)

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
        strategy_text = "\n".join(f"• {s}" for s in data["strategies"])
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
        pdf.multi_cell(0, 8, f"- {strategy}")

    # Section: Phaseout Thresholds
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Phaseout/Threshold Alerts", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for flag in payload["threshold_flags"]:
        pdf.multi_cell(0, 8, f"⚠️ {flag}")

    # Footer
    pdf.set_y(-30)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 10, "Valhalla Tax Services | www.valhallataxservice.com | (623) 887-7921", ln=True, align="C")

    return pdf.output(dest="S").encode("latin-1")
