from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class YearEndInput(BaseModel):
    filing_status: str
    w2_income: float
    business_income: float = 0
    capital_gains: float = 0
    itemized_deductions: float = 0
    retirement_contributions: float = 0
    hsa_contributions: float = 0
    estimated_payments: float = 0

@router.post("/year_end_plan")
def year_end_plan(data: YearEndInput):
    agi = data.w2_income + data.business_income + data.capital_gains
    taxable_income = agi - data.itemized_deductions - data.retirement_contributions - data.hsa_contributions
    estimated_tax = round(taxable_income * 0.22, 2)  # simple flat rate for estimate

    strategies = []
    today = datetime.today()

    if data.retirement_contributions < 23000:
        strategies.append("Max out 401(k) or traditional IRA before Dec 31.")
    if data.hsa_contributions < 8300:
        strategies.append("Max out HSA if eligible before Dec 31.")
    if data.capital_gains > 0:
        strategies.append("Harvest capital losses to offset gains by year-end.")
    if data.business_income > 0:
        strategies.append("Consider bonus equipment purchases for Section 179 deduction.")

    return {
        "agi": agi,
        "taxable_income": taxable_income,
        "estimated_tax": estimated_tax,
        "strategies": strategies,
        "year_end_deadline": "December 31, {}".format(today.year)
    }
