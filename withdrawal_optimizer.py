from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter()

class WithdrawalInput(BaseModel):
    age: int
    filing_status: str
    spending_needs: float
    brokerage_balance: float
    traditional_ira_balance: float
    roth_balance: float
    expected_return: float = 0.05
    tax_bracket: float = 0.22

class WithdrawalStrategy(BaseModel):
    year: int
    source: str
    amount: float
    reason: str

@router.post("/withdrawal_order_optimizer", response_model=List[WithdrawalStrategy])
def withdrawal_order_optimizer(data: WithdrawalInput):
    strategy = []
    age = data.age
    needs = data.spending_needs
    year = 2025

    # Phase 1: Use taxable first (assumes some basis to reduce tax drag)
    if data.brokerage_balance >= needs:
        strategy.append(WithdrawalStrategy(
            year=year,
            source="Taxable Account",
            amount=needs,
            reason="Use taxable assets first for basis harvesting and LTCG rates"
        ))
    else:
        if data.brokerage_balance > 0:
            strategy.append(WithdrawalStrategy(
                year=year,
                source="Taxable Account",
                amount=data.brokerage_balance,
                reason="Partial use of taxable assets"
            ))
            needs -= data.brokerage_balance

        # Phase 2: Use Traditional IRA up to bracket cap
        ira_limit = (80000 if data.filing_status == 'married_filing_jointly' else 40000)
        ira_withdrawal = min(needs, ira_limit)
        if ira_withdrawal > 0:
            strategy.append(WithdrawalStrategy(
                year=year,
                source="Traditional IRA",
                amount=ira_withdrawal,
                reason="Fill lower tax bracket with Traditional IRA withdrawals"
            ))
            needs -= ira_withdrawal

        # Phase 3: Use Roth last
        if needs > 0 and data.roth_balance >= needs:
            strategy.append(WithdrawalStrategy(
                year=year,
                source="Roth IRA",
                amount=needs,
                reason="Use Roth for tax-free income once other sources are depleted"
            ))

    return strategy
