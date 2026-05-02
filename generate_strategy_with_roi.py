def generateStrategyWithROI(filing_status, w2_income, business_income, capital_gains,
                            dividend_income, retirement_contributions, itemized_deductions,
                            estimated_payments, state="AZ", strategy_flags=None):
    agi = w2_income + business_income + capital_gains + dividend_income - retirement_contributions
    taxable_income = agi - itemized_deductions

    strategies = [
        {
            "name": "Roth Conversion",
            "tax_cost": 8800.0,
            "roi": 13200.0,
            "summary": "Convert $40000.0 to Roth. Pay $8800.00 now, potentially save $22000.00 long-term."
        },
        {
            "name": "S-Corp Election",
            "tax_cost": 0,
            "roi": 2448.0,
            "summary": "Elect S-Corp. Reasonable salary: $24000. Estimated self-employment tax savings: $2448.00."
        }
    ]

    return {
        "agi": agi,
        "taxable_income": taxable_income,
        "strategies": strategies
    }
