# report_generator.py

from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile
import os
import re

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

        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(30, 30, 30)
        self.set_text_color(255, 255, 255)

        self.cell(0, 10, safe_text(title), ln=True, fill=True)

        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 11)

        self.ln(3)

        for line in str(content).split("\n"):

            self.multi_cell(0, 7, safe_text(line))

        self.ln(5)

    def add_chart_image(self, img_path):
        self.ln(10)
        self.image(img_path, x=30, w=220)
        self.ln(5)

def add_kpi_card(self, title, value, x, y, w=55, h=28):

    # Card background
    self.set_fill_color(245, 245, 245)

    # Border
    self.set_draw_color(180, 180, 180)

    self.rect(x, y, w, h, style="DF")

    # KPI Title
    self.set_xy(x, y + 4)
    self.set_font("Helvetica", "B", 9)
    self.set_text_color(80, 80, 80)

    self.cell(w, 5, safe_text(title), align="C")

    # KPI Value
    self.set_xy(x, y + 12)
    self.set_font("Helvetica", "B", 16)
    self.set_text_color(0, 0, 0)

    self.cell(w, 8, safe_text(str(value)), align="C")

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

def generate_valhalla_report_v2_pdf(data: dict, logo_path=None) -> bytes:
    pdf = TaxReportPDF(orientation="P", unit="mm", format="Letter")
    pdf.logo_path = logo_path
    pdf.title = "Valhalla Tax Services - Client Tax Strategy Report"
    pdf.set_auto_page_break(auto=True, margin=15)

    # PAGE 1
    pdf.add_page()
    
    pdf.add_kpi_card(
        "AGI",
        f"${data.get('agi',0):,.0f}",
        15,
        45
    )

    pdf.add_kpi_card(
        "Federal Tax",
        f"${data.get('total_tax',0):,.0f}",
        75,
        45
    )

    pdf.add_kpi_card(
        "Refund",
        f"${data.get('refund',0):,.0f}",
        135,
        45
    )

    pdf.add_kpi_card(
        "Tax Score",
        data.get('tax_efficiency_score', "N/A"),
        195,
        45
    )
    executive_dashboard = f"""
Client: {data.get('client_name', '')}
Tax Year: {data.get('tax_year', '')}
Filing Status: {data.get('filing_status', '')}

Tax Efficiency Score: {data.get('tax_efficiency_score', '')}
Tax Opportunity Index: {data.get('tax_opportunity_index', '')}
Current Federal Bracket: {data.get('current_federal_bracket', '')}
Refund or Balance Position: {data.get('refund_or_balance_position', '')}

Top 3 Priority Actions:
1. {data.get('priority_1', '')}
2. {data.get('priority_2', '')}
3. {data.get('priority_3', '')}
"""

    pdf.add_section("Executive Dashboard", executive_dashboard)

    # PAGE 2
    pdf.add_page()

    advisor_summary = f"""
Current Situation:
{data.get('current_situation', '')}

What We Found:
{data.get('what_we_found', '')}

Primary Planning Message:
{data.get('primary_planning_message', '')}

Advisor Conclusion:
{data.get('advisor_conclusion', '')}
"""

    pdf.add_section("Advisor Summary", advisor_summary)

    # PAGE 3
    pdf.add_page()

    executive_summary = f"""
Confirmed Tax Position:
{data.get('confirmed_tax_position', '')}

Current Position Analysis:
{data.get('current_position_analysis', '')}

Advisor Conclusions:
{data.get('advisor_conclusions', '')}

Key Recommendations:
{data.get('key_recommendations', '')}
"""

    pdf.add_section("Executive Summary", executive_summary)

    # PAGE 4
    pdf.add_page()

    federal_analysis = f"""
W-2 Income: ${data.get('w2_income', 0):,.0f}
Schedule C Income/Loss: ${data.get('schedule_c_net_profit', 0):,.0f}
Adjusted Gross Income: ${data.get('agi', 0):,.0f}
Taxable Income: ${data.get('taxable_income', 0):,.0f}
Total Federal Tax: ${data.get('total_tax', 0):,.0f}
Federal Withholding: ${data.get('federal_withholding', 0):,.0f}
Refund: ${data.get('refund', 0):,.0f}
Balance Due: ${data.get('balance_due', 0):,.0f}

Advisor Commentary:
{data.get('federal_advisor_commentary', '')}
"""

    pdf.add_section("Federal Tax Analysis", federal_analysis)

    # PAGE 5
    pdf.add_page()

    business_analysis = f"""
Gross Receipts: ${data.get('schedule_c_gross_revenue', 0):,.0f}
Net Profit/Loss: ${data.get('schedule_c_net_profit', 0):,.0f}

Mileage Review:
{data.get('mileage_review', '')}

Documentation Review:
{data.get('documentation_review', '')}

Risk Assessment:
{data.get('risk_assessment', '')}
"""

    pdf.add_section("Business Analysis", business_analysis)

    # PAGE 6
    pdf.add_page()

    roth_strategy = f"""
Roth Opportunity Analysis:
{data.get('roth_opportunity_analysis', '')}

1-Year Plan:
{data.get('roth_1_year_plan', '')}

3-Year Plan:
{data.get('roth_3_year_plan', '')}

5-Year Plan:
{data.get('roth_5_year_plan', '')}
"""

    pdf.add_section("Retirement and Roth Strategy", roth_strategy)

    # PAGE 7
    pdf.add_page()

    action_plan = f"""
Next 30 Days:
{data.get('next_30_days', '')}

Next 90 Days:
{data.get('next_90_days', '')}

Before Year-End:
{data.get('before_year_end', '')}

Annual Review:
{data.get('annual_review', '')}
"""

    pdf.add_section("Action Plan Timeline", action_plan)

    # PAGE 8
    pdf.add_page()

    final_recommendation = f"""
Highest Priority Actions:
{data.get('highest_priority_actions', '')}

Monitor Annually:
{data.get('monitor_annually', '')}

Long-Term Tax Savings Potential:
{data.get('long_term_tax_savings_potential', '')}

Recommended Follow-Up Date:
{data.get('recommended_follow_up_date', '')}

Advisor Signature:
Valhalla Tax Services LLP
"""

    pdf.add_section("Final Advisor Recommendation", final_recommendation)

    return pdf.output(dest="S").encode("latin1")

