# arizona_tax.py

def calculate_arizona_tax(agi: float, filing_status: str) -> dict:
    """
    Arizona 2025 flat tax: 2.5% of taxable income.
    Applies standard deduction based on filing status.
    """
    standard_deductions = {
        "single": 13350,
        "married_filing_jointly": 26700,
        "head_of_household": 20025
    }
    deduction = standard_deductions.get(filing_status, 13350)
    az_taxable_income = max(agi - deduction, 0)
    tax_due = round(az_taxable_income * 0.025, 2)

    return {
        "state": "AZ",
        "agi": agi,
        "standard_deduction": deduction,
        "taxable_income": az_taxable_income,
        "tax_due": tax_due,
        "rate": "2.5%"
    }
