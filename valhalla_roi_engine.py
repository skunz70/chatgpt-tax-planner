from typing import Dict, Any, List


def _num(value: Any, default: float = 0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "").strip()
        return float(value)
    except Exception:
        return float(default)


def _money(value: Any) -> str:
    return f"${_num(value):,.0f}"


def generate_roi_analysis(client_data: Dict) -> Dict:
    data = client_data or {}

    schedule_c_profit = _num(data.get("schedule_c_net_profit", 0))
    contract_labor = _num(data.get("contract_labor", 0))
    capital_gains = _num(data.get("capital_gains", 0))
    total_tax = _num(data.get("total_tax", 0))
    federal_withholding = _num(data.get("federal_withholding", 0))

    items: List[Dict] = []

    if schedule_c_profit >= 75000:
        value = round(schedule_c_profit * 0.055, 0)
        items.append({
            "strategy": "S-Corporation Feasibility",
            "estimated_value": _money(value),
            "score": 88,
            "difficulty": "Medium",
            "timeline": "Within 90 days",
            "basis": "Estimated SE tax optimization opportunity before payroll/entity costs."
        })

    if schedule_c_profit > 25000:
        value = round(schedule_c_profit * 0.20, 0)
        items.append({
            "strategy": "Retirement Contribution Optimization",
            "estimated_value": _money(value),
            "score": 90,
            "difficulty": "Low to Medium",
            "timeline": "Current tax year",
            "basis": "Estimated tax value from deductible retirement contribution planning."
        })

    if contract_labor > 50000:
        items.append({
            "strategy": "Contractor Compliance Review",
            "estimated_value": "$2,500-$15,000 risk protection value",
            "score": 86,
            "difficulty": "Medium",
            "timeline": "Immediate",
            "basis": "Risk control for 1099, worker classification, and deduction substantiation."
        })

    if capital_gains > 0:
        value = round(capital_gains * 0.15, 0)
        items.append({
            "strategy": "Capital Gain Bracket Management",
            "estimated_value": _money(value),
            "score": 74,
            "difficulty": "Low",
            "timeline": "Before year-end",
            "basis": "Estimated value from gain/loss timing and bracket-aware harvesting."
        })

    balance = total_tax - federal_withholding
    if balance > 0:
        items.append({
            "strategy": "Withholding / Estimated Payment Correction",
            "estimated_value": _money(balance),
            "score": 80,
            "difficulty": "Low",
            "timeline": "Immediate",
            "basis": "Cash-flow correction based on supplied tax and withholding data."
        })

    items = sorted(items, key=lambda x: x.get("score", 0), reverse=True)

    return {
        "roi_items": items,
        "top_roi_strategy": items[0] if items else None,
        "summary": "ROI estimates are planning illustrations based on supplied client facts and should be modeled before implementation."
    }