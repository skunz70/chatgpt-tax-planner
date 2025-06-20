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
