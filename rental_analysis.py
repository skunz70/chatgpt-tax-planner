from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class RentalPropertyInput(BaseModel):
    rental_income: float
    expenses: float
    mortgage_interest: float
    property_tax: float
    insurance: float
    repairs: float = 0
    purchase_price: float
    land_value: float
    filing_status: str
    active_participation: bool = False

@router.post("/rental_analysis")
def rental_analysis(data: RentalPropertyInput):
    depreciation_basis = data.purchase_price - data.land_value
    annual_depreciation = round(depreciation_basis / 27.5, 2)

    total_expenses = (
        data.expenses + data.mortgage_interest +
        data.property_tax + data.insurance + data.repairs + annual_depreciation
    )

    taxable_income = round(data.rental_income - total_expenses, 2)
    cash_flow = round(data.rental_income - (
        data.expenses + data.mortgage_interest + data.property_tax + data.insurance + data.repairs
    ), 2)

    passive_loss_warning = None
    if taxable_income < 0:
        if data.active_participation and data.filing_status == "married_filing_jointly":
            passive_loss_warning = "Passive loss may be limited to $25,000 depending on AGI."
        else:
            passive_loss_warning = "Passive loss may not be deductible this year."

    return {
        "rental_income": data.rental_income,
        "cash_flow": cash_flow,
        "taxable_income": taxable_income,
        "annual_depreciation": annual_depreciation,
        "passive_loss_warning": passive_loss_warning
    }
