from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ScheduleCInput(BaseModel):
    gross_income: float
    expenses: float
    home_office_expense: float = 0
    vehicle_expense: float = 0
    retirement_contribution: float = 0

@router.post("/schedule_c_analysis")
def schedule_c_analysis(data: ScheduleCInput):
    net_profit = data.gross_income - data.expenses - data.home_office_expense - data.vehicle_expense
    se_tax = round(net_profit * 0.153, 2) if net_profit > 0 else 0
    optimized_profit = net_profit - data.retirement_contribution

    suggestions = []
    if data.home_office_expense == 0:
        suggestions.append("Consider calculating and claiming a home office deduction.")
    if data.vehicle_expense == 0:
        suggestions.append("Evaluate business mileage or actual vehicle expense deduction.")
    if data.retirement_contribution == 0:
        suggestions.append("Consider contributing to a SEP IRA or Solo 401(k) to reduce net income.")

    return {
        "gross_income": data.gross_income,
        "expenses": data.expenses,
        "net_profit": round(net_profit, 2),
        "self_employment_tax": se_tax,
        "optimized_profit_after_retirement": round(optimized_profit, 2),
        "suggestions": suggestions
    }
