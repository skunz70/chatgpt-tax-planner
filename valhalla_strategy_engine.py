from typing import Dict, List


def _num(value, default=0):
    try:
        return float(value)
    except Exception:
        return float(default)


class StrategyEngine:
    def __init__(self, client_data: Dict):
        self.data = client_data or {}
        self.agi = _num(self.data.get("agi", 0))
        self.taxable_income = _num(self.data.get("taxable_income", 0))
        self.total_tax = _num(self.data.get("total_tax", 0))
        self.schedule_c_profit = _num(self.data.get("schedule_c_net_profit", 0))
        self.capital_gains = _num(self.data.get("capital_gains", 0))
        self.w2_income = _num(self.data.get("w2_income", 0))
        self.age = _num(self.data.get("age", 0))
        self.has_marketplace = bool(self.data.get("aca_marketplace", False))
        self.has_retirement = bool(self.data.get("has_retirement_accounts", False))

    def build_priority_actions(self) -> List[Dict]:
        actions = []

        if self.schedule_c_profit > 0:
            actions.append({
                "priority": "High",
                "title": "Schedule C Documentation Optimization",
                "estimated_savings": "$3,000-$10,000 risk protection value",
                "timeline": "Immediate",
                "reason": "Protects existing deductions and reduces audit exposure.",
            })

        if self.schedule_c_profit >= 60000:
            actions.append({
                "priority": "High",
                "title": "Evaluate S-Corporation Election",
                "estimated_savings": "$5,000-$15,000 annually",
                "timeline": "Within 90 days",
                "reason": "Self-employment tax exposure becomes material at higher profitability levels.",
            })

        if self.schedule_c_profit > 0 or self.w2_income > 0:
            actions.append({
                "priority": "High",
                "title": "Maximize Retirement Contributions",
                "estimated_savings": "$2,000-$12,000 annually",
                "timeline": "Current tax year",
                "reason": "Improves tax efficiency while accelerating long-term wealth accumulation.",
            })

        if self.capital_gains > 0:
            actions.append({
                "priority": "Medium",
                "title": "Capital Gain Bracket Management",
                "estimated_savings": "$1,500-$8,000",
                "timeline": "Before year-end",
                "reason": "Strategic harvesting may reduce future capital gains exposure.",
            })

        if self.has_retirement and self.taxable_income < 250000:
            actions.append({
                "priority": "Medium",
                "title": "Roth Conversion Window Analysis",
                "estimated_savings": "Long-term tax reduction potential",
                "timeline": "Current or future low-income years",
                "reason": "Lower bracket years may create favorable Roth conversion opportunities.",
            })

        if self.age >= 63:
            actions.append({
                "priority": "Medium",
                "title": "IRMAA and Medicare Threshold Planning",
                "estimated_savings": "$1,000-$6,000",
                "timeline": "Multi-year planning",
                "reason": "Managing MAGI may reduce future Medicare premium surcharges.",
            })

        if self.has_marketplace:
            actions.append({
                "priority": "High",
                "title": "ACA Premium Credit Monitoring",
                "estimated_savings": "$2,000-$15,000",
                "timeline": "Quarterly",
                "reason": "Income swings may create subsidy repayment exposure.",
            })

        if self.total_tax > 0 and self.taxable_income > 0:
            actions.append({
                "priority": "Medium",
                "title": "Withholding and Estimated Tax Calibration",
                "estimated_savings": "Penalty and cash-flow optimization",
                "timeline": "Immediate",
                "reason": "Improves predictability and reduces underpayment risk.",
            })

        return actions[:7]

    def build_dynamic_sections(self) -> List[Dict]:
        sections = []

        if self.schedule_c_profit > 0:
            sections.append({
                "section": "Business Owner Strategy",
                "body": "Current planning should focus on documentation quality, retirement integration, contractor compliance, and long-term entity transition readiness. The business currently produces self-employment tax exposure that may eventually justify payroll optimization and S-Corp planning."
            })

        if self.schedule_c_profit >= 60000:
            sections.append({
                "section": "S-Corporation Feasibility",
                "body": "Current profitability appears high enough to begin formal S-Corporation modeling. A detailed reasonable compensation analysis should be completed before implementation."
            })

        if self.capital_gains > 0:
            sections.append({
                "section": "Capital Gain Planning",
                "body": "The return includes capital gain activity. Future planning should monitor gain recognition timing, tax-loss harvesting opportunities, and long-term capital gain bracket utilization."
            })

        if self.has_retirement:
            sections.append({
                "section": "Retirement Tax Planning",
                "body": "Retirement accounts create opportunities for tax bracket management, Roth conversion planning, and future required minimum distribution mitigation."
            })

        if self.age >= 63:
            sections.append({
                "section": "Medicare and IRMAA Planning",
                "body": "Future income levels should be monitored carefully to manage Medicare premium surcharge thresholds and optimize long-term retirement cash flow."
            })

        return sections


def generate_dynamic_tax_strategy(client_data: Dict) -> Dict:
    engine = StrategyEngine(client_data)
    return {
        "priority_actions": engine.build_priority_actions(),
        "dynamic_sections": engine.build_dynamic_sections(),
    }


if __name__ == "__main__":
    sample = {
        "agi": 125000,
        "taxable_income": 92000,
        "total_tax": 14000,
        "schedule_c_net_profit": 85000,
        "capital_gains": 12000,
        "has_retirement_accounts": True,
        "age": 64,
    }

    import json
    print(json.dumps(generate_dynamic_tax_strategy(sample), indent=2))