from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class CapGainsInput(BaseModel):
    filing_status: str
    ordinary_income: float
    capital_gains: float

@router.post("/cap_gain_projection")
def cap_gain_projection(data: CapGainsInput):
    # 2025 capital gains thresholds (approximate)
    thresholds = {
        "single": {"0%": 44725, "15%": 492300},
        "married_filing_jointly": {"0%": 89450, "15%": 553850},
        "head_of_household": {"0%": 59750, "15%": 523050}
    }

    brackets = thresholds.get(data.filing_status.lower())
    if not brackets:
        return {"error": "Invalid filing status"}

    total_income = data.ordinary_income + data.capital_gains

    if total_income <= brackets["0%"]:
        rate = "0%"
    elif total_income <= brackets["15%"]:
        rate = "15%"
    else:
        rate = "20%"

    # Approximate tax on capital gains
    rates = {"0%": 0.00, "15%": 0.15, "20%": 0.20}
    est_tax = round(data.capital_gains * rates[rate], 2)

    return {
        "filing_status": data.filing_status,
        "ordinary_income": data.ordinary_income,
        "capital_gains": data.capital_gains,
        "estimated_tax": est_tax,
        "capital_gains_rate": rate
    }
