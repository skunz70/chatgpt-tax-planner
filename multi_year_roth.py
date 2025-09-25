from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class MultiYearRothRequest(BaseModel):
    current_agi: float
    annual_conversion: float
    years: int
    filing_status: str

class YearlyProjection(BaseModel):
    year: int
    projected_agi: float
    estimated_tax: float
    marginal_rate: str

@router.post("/multi_year_roth_projection", response_model=List[YearlyProjection])
def compare_scenarios(data: MultiYearRothRequest):

    projections = []
    for i in range(data.years):
        year = 2025 + i
        new_agi = data.current_agi + data.annual_conversion * (i + 1)
        tax = new_agi * 0.24  # Simplified
        rate = "24%"
        projections.append(YearlyProjection(
            year=year,
            projected_agi=new_agi,
            estimated_tax=round(tax, 2),
            marginal_rate=rate
        ))
    return projections
