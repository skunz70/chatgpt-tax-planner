from fpdf import FPDF
import re


def _safe_text(value):
    text = "" if value is None else str(value)
    replacements = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "•": "-", "→": "->", "·": "-"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.encode("latin-1", "ignore").decode("latin-1")


def _money(value):
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"


class ValhallaPremiumPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 7, "Valhalla Tax Services | Comprehensive Tax Strategy Report", 0, 1, "L")
        self.set_draw_color(180, 180, 180)
        self.line(10, 16, 269, 16)
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()} | Valhalla Tax Services", 0, 0, "C")

    def section_title(self, title):
        self.ln(4)
        self.set_fill_color(90, 90, 90)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 9, _safe_text(title), 0, 1, "L", True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def body_text(self, text, size=10):
        self.set_font("Helvetica", "", size)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, _safe_text(text))
        self.ln(1)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, _safe_text(f"- {text}"))

    def callout_box(self, title, body):
        x = self.get_x(); y = self.get_y(); w = 260
        self.set_fill_color(245, 245, 245)
        self.set_draw_color(170, 170, 170)
        self.rect(x, y, w, 29, "DF")
        self.set_xy(x + 4, y + 4)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(80, 80, 80)
        self.cell(w - 8, 6, _safe_text(title), 0, 1)
        self.set_x(x + 4)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(w - 8, 5, _safe_text(body))
        self.set_y(y + 33)


def generate_valhalla_premium_tax_report(data: dict, output_path: str = "valhalla_premium_report.pdf"):
    data = data or {}
    client = data.get("client_name", "Client")
    tax_year = data.get("tax_year", "2025")
    filing_status = data.get("filing_status", "Head of Household")
    dependents = data.get("dependents", 2)
    agi = data.get("agi", 24266)
    taxable_income = data.get("taxable_income", 0)
    total_tax = data.get("total_tax", 3429)
    refund = data.get("refund", 6731)
    gross_revenue = data.get("schedule_c_gross_revenue", 216265)
    net_profit = data.get("schedule_c_net_profit", 24266)
    contract_labor = data.get("contract_labor", 107920)
    depreciation = data.get("depreciation", 16624)
    vehicle_deduction = data.get("vehicle_deduction", 4628)
    qbi_deduction = data.get("qbi_deduction", 4510)

    pdf = ValhallaPremiumPDF(orientation="L", unit="mm", format="Letter")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_fill_color(70, 70, 70)
    pdf.rect(0, 0, 280, 34, "F")
    pdf.set_xy(12, 10)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 9, "VALHALLA TAX SERVICES", 0, 1)
    pdf.set_x(12)
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 8, "Comprehensive Tax Strategy Report", 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(12)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, f"Client: {_safe_text(client)}", 0, 1)
    pdf.cell(0, 8, f"Tax Year: {tax_year}", 0, 1)
    pdf.cell(0, 8, "Prepared by: Scott Kunz, ChFC, TPCP, Enrolled Agent", 0, 1)
    pdf.ln(6)
    pdf.callout_box("Executive Summary", "This return reflects a Head of Household taxpayer with two dependents and Schedule C self-employment income. The current return shows no federal income tax after deductions and credits; the primary tax burden is self-employment tax. The planning opportunity is to move from expense suppression to structured tax optimization.")

    pdf.section_title("Confirmed Tax Data Summary")
    rows = [("Filing Status", filing_status), ("Dependents", dependents), ("Adjusted Gross Income", _money(agi)), ("Taxable Income", _money(taxable_income)), ("Total Tax", _money(total_tax)), ("Refund", _money(refund))]
    col_w = [70, 55, 70, 55]
    for i in range(0, len(rows), 2):
        left = rows[i]; right = rows[i + 1] if i + 1 < len(rows) else ("", "")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_fill_color(245, 245, 245)
        pdf.cell(col_w[0], 8, _safe_text(left[0]), 1, 0, "L", True)
        pdf.cell(col_w[1], 8, _safe_text(left[1]), 1, 0, "L")
        pdf.cell(col_w[2], 8, _safe_text(right[0]), 1, 0, "L", True)
        pdf.cell(col_w[3], 8, _safe_text(right[1]), 1, 1, "L")

    pdf.section_title("Federal Tax Analysis")
    pdf.body_text("The client currently has zero taxable income after the standard deduction. Traditional income tax reduction strategies have limited immediate value at the current income level. The tax liability is primarily self-employment tax, which makes entity structure, retirement planning, child employment, and business deduction optimization the highest-value planning categories.")
    pdf.section_title("Arizona Tax Analysis")
    pdf.body_text("Arizona exposure appears minimal based on the low taxable income profile. The more significant planning opportunity remains at the federal level through Schedule C and self-employment tax management.")
    pdf.section_title("Schedule C Business Analysis")
    try:
        expense_ratio = (float(gross_revenue) - float(net_profit)) / float(gross_revenue) * 100
    except Exception:
        expense_ratio = 0
    pdf.body_text(f"The business generated gross revenue of {_money(gross_revenue)} and net profit of {_money(net_profit)}, producing an approximate expense ratio of {expense_ratio:.1f}%. Major deductions include contract labor of {_money(contract_labor)}, depreciation of {_money(depreciation)}, and vehicle deductions of {_money(vehicle_deduction)}.")
    pdf.bullet("High contract labor should be reviewed for 1099 vs W-2 classification risk.")
    pdf.bullet("Vehicle strategy should be reviewed annually: mileage method vs actual expense / Section 179 planning.")
    pdf.bullet(f"QBI deduction currently modeled at {_money(qbi_deduction)}.")

    pdf.add_page()
    pdf.section_title("Strategy Savings Summary")
    strategies = [["Strategy", "Estimated Savings", "Timing"], ["S-Corp Election", "$5,000 - $7,500", "Future, once profit supports payroll"], ["Retirement Contributions", "$4,000 - $8,000", "Current / annual"], ["Vehicle Strategy", "$12,000 - $18,000", "Year 1 potential"], ["Child Employment", "$6,000 - $8,500", "Annual, if valid work exists"], ["QBI Optimization", "$3,500 - $4,500", "As profit increases"]]
    widths = [75, 55, 125]
    for r, row in enumerate(strategies):
        pdf.set_font("Helvetica", "B" if r == 0 else "", 10)
        if r == 0:
            pdf.set_fill_color(90, 90, 90); pdf.set_text_color(255, 255, 255)
        else:
            pdf.set_fill_color(255, 255, 255); pdf.set_text_color(0, 0, 0)
        for i, cell in enumerate(row):
            pdf.cell(widths[i], 8, _safe_text(cell), 1, 0, "L", r == 0)
        pdf.ln()
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    pdf.body_text("These estimates are planning ranges, not guaranteed results. Actual savings depend on final income, business use percentages, payroll reasonableness, documentation quality, and implementation timing.")

    pdf.section_title("Top 3 Priority Actions")
    pdf.callout_box("1. Build a Quarterly Schedule C Tax Planning System", "Estimated impact: prevents surprise tax bills and identifies deduction gaps before year-end. This should be implemented immediately because the taxpayer's primary exposure is self-employment tax.")
    pdf.callout_box("2. Review Contractor Classification and Labor Documentation", "Estimated impact: risk reduction plus cleaner business records. Contract labor is the largest Schedule C expense category and should be documented carefully.")
    pdf.callout_box("3. Prepare for Entity and Retirement Strategy as Profit Scales", "Estimated impact: $5,000 to $15,000+ annually once profit reaches the appropriate level. S-Corp and retirement planning should be evaluated before year-end, not after filing season.")

    pdf.section_title("Do This Now Checklist")
    for item in ["Set up quarterly tax planning reviews.", "Confirm all contractor Forms W-9 are on file.", "Review whether any contractors should be treated as employees.", "Evaluate Solo 401(k) or SEP IRA options.", "Compare vehicle mileage method against actual expense method.", "Create a monthly profit tracking dashboard.", "Run an S-Corp break-even analysis if 2026 net profit trends above $60,000 to $80,000."]:
        pdf.bullet(item)

    pdf.section_title("Multi-Year Strategy Roadmap")
    pdf.body_text("0-90 Days: Clean up business records, implement quarterly planning, verify contractor documentation, and evaluate retirement account options.\n\n6-12 Months: Review profit trend, vehicle strategy, child employment feasibility, and estimated payment discipline.\n\n12-24 Months: If profitability increases, evaluate S-Corp election, payroll strategy, retirement contribution optimization, and QBI coordination.")
    pdf.section_title("Advisor Summary")
    pdf.body_text("The client is not currently in a high income-tax position; the main issue is self-employment tax and business structure. The best strategy is not simply to keep income low. The better long-term strategy is to grow business profit while using entity structure, retirement planning, documented deductions, and family-based planning where appropriate.")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(0, 6, _safe_text("Prepared by Scott Kunz, ChFC, TPCP, Enrolled Agent | Valhalla Tax Services"))
    pdf.output(output_path)
    return output_path


def demo_generate_valhalla_premium_tax_report():
    sample_data = {"client_name": "Nathan Deratany", "tax_year": 2025, "filing_status": "Head of Household", "dependents": 2, "agi": 24266, "taxable_income": 0, "total_tax": 3429, "refund": 6731, "schedule_c_gross_revenue": 216265, "schedule_c_net_profit": 24266, "contract_labor": 107920, "depreciation": 16624, "vehicle_deduction": 4628, "qbi_deduction": 4510}
    return generate_valhalla_premium_tax_report(sample_data, "valhalla_premium_report.pdf")
