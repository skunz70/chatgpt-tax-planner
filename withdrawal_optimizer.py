from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class WithdrawalRequest(BaseModel):
    age: int
    filing_status: str
    spending_needs: float
    brokerage_balance: float
    traditional_ira_balance: float
    roth_balance: float
    expected_return: float = 0.05
    tax_bracket: float = 0.22

class WithdrawalStrategy(BaseModel):
    source: str
    amount: float
    tax_impact: str

@router.post("/withdrawal_order_optimizer", response_model=List[WithdrawalStrategy])
def withdrawal_order_optimizer(data: WithdrawalRequest):
    strategies = []
    spending = data.spending_needs
    notes = []

    if data.brokerage_balance >= spending:
        strategies.append(WithdrawalStrategy(source="Brokerage", amount=spending, tax_impact="Capital gains"))
    elif data.brokerage_balance + data.traditional_ira_balance >= spending:
        strategies.append(WithdrawalStrategy(source="Brokerage", amount=data.brokerage_balance, tax_impact="Capital gains"))
        remaining = spending - data.brokerage_balance
        strategies.append(WithdrawalStrategy(source="Traditional IRA", amount=remaining, tax_impact="Ordinary income"))
    elif data.brokerage_balance + data.traditional_ira_balance + data.roth_balance >= spending:
        strategies.append(WithdrawalStrategy(source="Brokerage", amount=data.brokerage_balance, tax_impact="Capital gains"))
        strategies.append(WithdrawalStrategy(source="Traditional IRA", amount=data.traditional_ira_balance, tax_impact="Ordinary income"))
        remaining = spending - data.brokerage_balance - data.traditional_ira_balance
        strategies.append(WithdrawalStrategy(source="Roth IRA", amount=remaining, tax_impact="Tax-free"))

    return strategies
