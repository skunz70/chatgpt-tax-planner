from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class RothInput(BaseModel):
    current_agi: float
    conversion_amount: float
    filing_status: str

@router.post("/roth_projection")
def roth_projection(data: RothInput):
    brackets = {
        "single": [
            (0, 11000, 0.10),
            (11000, 44725, 0.12),
            (44725, 95375, 0.22),
            (95375, 182100, 0.24),
        ],
        "married_filing_jointly": [
            (0, 22000, 0.10),
            (22000, 89450, 0.12),
            (89450, 190750, 0.22),
            (190750, 364200, 0.24),
        ]
    }

    used_brackets = brackets.get(data.filing_status.lower(), [])
    remaining_conversion = data.conversion_amount
    taxable_income = data.current_agi
    tax_due = 0

    for lower, upper, rate in used_brackets:
        if taxable_income >= upper:
            continue
        bracket_room = upper - max(taxable_income, lower)
        convert_in_bracket = min(remaining_conversion, bracket_room)
        tax_due += convert_in_bracket * rate
        taxable_income += convert_in_bracket
        remaining_conversion -= convert_in_bracket
        if remaining_conversion <= 0:
            break

    return {
        "total_conversion": data.conversion_amount,
        "estimated_tax_due": round(tax_due, 2),
        "marginal_rate_used": f"{int(rate * 100)}%"
    }
