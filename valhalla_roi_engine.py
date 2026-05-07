from typing import Dict, List


FEDERAL_BRACKETS = {
    "single": 0.24,
    "mfj": 0.22,
    "married filing jointly": 0.22,
    "hoh": 0.22,
    "head of household": 0.22,
}

ARIZONA_RATE = 0.025


class ROIEngine:
    def __init__(self, client_data: Dict):
        self.data = client_data or {}
        self.filing_status = str(self.data.get("filing_status", "single")).lower()
        self.schedule_c_profit = self._num(self.data.get("schedule_c_net_profit", 0))
        self.taxable_income = self._num(self.data.get("taxable_income", 0))
        self.capital_gains = self._num(self.data.get("capital_gains", 0))
        self.age = self._num(self.data.get("age", 0))
        self.has_retirement = bool(self.data.get("has_retirement_accounts", False))

    def _num(self, value, default=0):
        try:
            return float(value)
        except Exception:
            return float(default)

    def marginal_rate(self):
        return FEDERAL_BRACKETS.get(self.filing_status, 0.22)

    def combined_rate(self):
        return self.marginal_rate() + ARIZONA_RATE

    def estimate_retirement_savings(self):
        if self.schedule_c_profit <= 0 and self.taxable_income <= 0:
            return None

        contribution_target = min(max(self.schedule_c_profit * 0.20, 5000), 30000)
        savings = contribution_target * self.combined_rate()

        return {
            "strategy": "Solo 401(k) / SEP IRA",
            "estimated_savings": round(savings, 0),
            "implementation_difficulty": "Low",
            "priority": "High",
            "timeline": "Current Year",
            "advisor_reasoning": "Current income supports additional retirement contribution capacity with immediate federal and Arizona tax benefits.",
        }

    def estimate_scorp_savings(self):
        if self.schedule_c_profit < 60000:
            return None

        reasonable_salary = self.schedule_c_profit * 0.50
        se_tax_savings = (self.schedule_c_profit - reasonable_salary) * 0.153 * 0.60

        return {
            "strategy": "S-Corporation Election",
            "estimated_savings": round(se_tax_savings, 0),
            "implementation_difficulty": "Medium",
            "priority": "High",
            "timeline": "Within 12 Months",
            "advisor_reasoning": "Current profitability suggests self-employment tax exposure is becoming material enough to justify formal entity modeling.",
        }

    def estimate_capital_gain_strategy(self):
        if self.capital_gains <= 0:
            return None

        estimated_savings = self.capital_gains * 0.15

        return {
            "strategy": "Capital Gain Optimization",
            "estimated_savings": round(estimated_savings, 0),
            "implementation_difficulty": "Low",
            "priority": "Medium",
            "timeline": "Before Year-End",
            "advisor_reasoning": "Strategic harvesting and bracket management may improve long-term capital gain efficiency.",
        }

    def estimate_irmaa_strategy(self):
        if self.age < 63:
            return None

        return {
            "strategy": "IRMAA Threshold Management",
            "estimated_savings": 2400,
            "implementation_difficulty": "Medium",
            "priority": "Medium",
            "timeline": "Multi-Year",
            "advisor_reasoning": "Managing future Medicare premium thresholds may create meaningful retirement cash-flow improvements.",
        }

    def estimate_roth_strategy(self):
        if not self.has_retirement:
            return None

        projected_value = max(self.taxable_income * 0.08, 3000)

        return {
            "strategy": "Roth Conversion Analysis",
            "estimated_savings": round(projected_value, 0),
            "implementation_difficulty": "Medium",
            "priority": "Medium",
            "timeline": "Low-Income Years",
            "advisor_reasoning": "Current tax brackets may support partial Roth conversion opportunities with long-term tax diversification benefits.",
        }

    def build_roi_strategies(self) -> List[Dict]:
        strategies = [
            self.estimate_retirement_savings(),
            self.estimate_scorp_savings(),
            self.estimate_capital_gain_strategy(),
            self.estimate_irmaa_strategy(),
            self.estimate_roth_strategy(),
        ]

        strategies = [s for s in strategies if s]

        for strategy in strategies:
            strategy["score"] = self.calculate_strategy_score(strategy)

        strategies.sort(key=lambda x: x["score"], reverse=True)

        return strategies

    def calculate_strategy_score(self, strategy: Dict):
        savings = strategy.get("estimated_savings", 0)
        difficulty = strategy.get("implementation_difficulty", "Medium")
        priority = strategy.get("priority", "Medium")

        score = 50

        score += min(savings / 250, 30)

        if difficulty == "Low":
            score += 10
        elif difficulty == "Medium":
            score += 5

        if priority == "High":
            score += 10
        elif priority == "Medium":
            score += 5

        return round(min(score, 99))

    def tax_efficiency_score(self):
        score = 100

        if self.schedule_c_profit > 0:
            score -= 12

        if self.schedule_c_profit >= 60000:
            score -= 10

        if not self.has_retirement:
            score -= 15

        if self.capital_gains > 0:
            score -= 6

        score = max(score, 45)

        optimized = min(score + 25, 96)

        return {
            "current_score": score,
            "optimized_score": optimized,
            "advisor_summary": "The current tax structure contains identifiable planning opportunities that may improve long-term tax efficiency, cash-flow control, and wealth accumulation if implemented properly.",
        }


def generate_roi_analysis(client_data: Dict):
    engine = ROIEngine(client_data)

    return {
        "tax_efficiency": engine.tax_efficiency_score(),
        "roi_strategies": engine.build_roi_strategies(),
    }


if __name__ == "__main__":
    sample = {
        "filing_status": "MFJ",
        "schedule_c_net_profit": 85000,
        "taxable_income": 92000,
        "capital_gains": 12000,
        "age": 64,
        "has_retirement_accounts": True,
    }

    import json
    print(json.dumps(generate_roi_analysis(sample), indent=2))