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

        filing_status = self.data.get("filing_status", "MFJ")
        federal_withholding = _num(
            self.data.get("federal_withholding", 0)
        )

        # ---- Federal brackets ----

        if filing_status == "MFJ":
            top_22_bracket = 206700
        else:
            top_22_bracket = 103350

        remaining_22_room = max(
            0,
            top_22_bracket - self.taxable_income
        )

        # ---- Schedule C documentation ----

        if self.schedule_c_profit > 0:

            actions.append({
                "priority": "High",
                "title": "Schedule C Documentation Optimization",
                "estimated_savings": "$3,000-$10,000 risk protection value",
                "timeline": "Immediate",
                "reason": "Protects deductions, strengthens substantiation, and reduces audit exposure.",
            })

        # ---- S-Corp planning ----

        if self.schedule_c_profit >= 80000:

            estimated_se_tax_savings = round(
                self.schedule_c_profit * 0.08,
                0
            )

            actions.append({
                "priority": "High",
                "title": "S-Corporation Election Analysis",
                "estimated_savings": f"${estimated_se_tax_savings:,.0f}",
                "timeline": "Within 12 months",
                "reason": "Current profitability may justify payroll optimization and self-employment tax reduction planning.",
            })

        # ---- Retirement planning ----

        if self.schedule_c_profit > 25000 or self.w2_income > 0:

            estimated_retirement_savings = round(
                self.schedule_c_profit * 0.22,
                0
            )

            actions.append({
                "priority": "High",
                "title": "Solo 401(k) / Retirement Contribution Optimization",
                "estimated_savings": f"${estimated_retirement_savings:,.0f}",
                "timeline": "Current tax year",
                "reason": "Current earned income creates meaningful pre-tax retirement contribution opportunities.",
            })

        # ---- Roth planning ----

        if (
            self.has_retirement
            and remaining_22_room > 25000
            and self.age >= 59
        ):

            roth_capacity = min(
                remaining_22_room,
                50000
            )

            actions.append({
                "priority": "Medium",
                "title": "Partial Roth Conversion Planning",
                "estimated_savings": f"${round(roth_capacity * 0.15, 0):,.0f}",
                "timeline": "Low-income or bracket-management years",
                "reason": f"Approximately ${remaining_22_room:,.0f} remains before entering the next federal bracket.",
            })

        # ---- Capital gains ----

        if self.capital_gains > 0:

            actions.append({
                "priority": "Medium",
                "title": "Capital Gain Bracket Management",
                "estimated_savings": f"${round(self.capital_gains * 0.15, 0):,.0f}",
                "timeline": "Before year-end",
                "reason": "Future gain harvesting should be coordinated with ordinary income and bracket thresholds.",
            })

        # ---- IRMAA ----

        if self.age >= 63 and self.agi > 200000:

            actions.append({
                "priority": "Medium",
                "title": "IRMAA Threshold Planning",
                "estimated_savings": "$2,500+",
                "timeline": "Multi-year planning",
                "reason": "Future Medicare premium surcharges may become material at current projected income levels.",
            })

        # ---- Withholding correction ----

        if self.total_tax > federal_withholding:

            projected_balance_due = round(
                self.total_tax - federal_withholding,
                0
            )

            actions.append({
                "priority": "High",
                "title": "Federal / Arizona Withholding Correction",
                "estimated_savings": f"${projected_balance_due:,.0f}",
                "timeline": "Immediate",
                "reason": "Projected underwithholding should be corrected proactively to improve cash flow predictability and avoid penalties.",
            })

        return actions[:7]

    def build_dynamic_sections(self) -> List[Dict]:

        sections = []

        if self.schedule_c_profit > 0:

            sections.append({
                "section": "Business Owner Strategy",
                "body": "Current planning should focus on documentation quality, retirement integration, contractor compliance, and long-term entity transition readiness."
            })

        if self.schedule_c_profit >= 80000:

            sections.append({
                "section": "S-Corporation Feasibility",
                "body": "Current profitability appears high enough to begin formal S-Corporation modeling and reasonable compensation analysis."
            })

        if self.capital_gains > 0:

            sections.append({
                "section": "Capital Gain Planning",
                "body": "Capital gain activity should be coordinated with bracket management and future harvesting opportunities."
            })

        if self.has_retirement:

            sections.append({
                "section": "Retirement Tax Planning",
                "body": "Retirement accounts create opportunities for bracket management and future Roth conversion analysis."
            })

        if self.age >= 63:

            sections.append({
                "section": "Medicare and IRMAA Planning",
                "body": "Future Medicare premium thresholds should be monitored proactively."
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
        "agi": 185000,
        "taxable_income": 142000,
        "total_tax": 24000,
        "federal_withholding": 22800,
        "schedule_c_net_profit": 95000,
        "capital_gains": 18000,
        "has_retirement_accounts": True,
        "age": 64,
    }

    import json

    print(
        json.dumps(
            generate_dynamic_tax_strategy(sample),
            indent=2
        )
    )