# state_tax_data.py

state_tax_brackets = {
    "AZ": {
        "2025": [
            {"rate": 0.023, "bracket": 29375},
            {"rate": 0.027, "bracket": 58750},
            {"rate": 0.033, "bracket": 117500},
            {"rate": 0.041, "bracket": float("inf")}
        ]
    },
    "CA": {
        "2025": [
            {"rate": 0.01, "bracket": 10084},
            {"rate": 0.02, "bracket": 23876},
            {"rate": 0.04, "bracket": 37784},
            {"rate": 0.06, "bracket": 52421},
            {"rate": 0.08, "bracket": 66295},
            {"rate": 0.093, "bracket": 338639},
            {"rate": 0.103, "bracket": 406364},
            {"rate": 0.113, "bracket": 677275},
            {"rate": 0.123, "bracket": float("inf")}
        ]
    },
    "TX": {
        "2025": []  # Texas has no state income tax
    }
}

def calculate_state_tax(income: float, state: str, year: str = "2025") -> float:
    brackets = state_tax_brackets.get(state.upper(), {}).get(year, [])
    if not brackets:
        return 0.0  # Assume no state tax

    tax = 0.0
    prev_limit = 0
    for bracket in brackets:
        if income > bracket["bracket"]:
            tax += (bracket["bracket"] - prev_limit) * bracket["rate"]
            prev_limit = bracket["bracket"]
        else:
            tax += (income - prev_limit) * bracket["rate"]
            break
    return round(tax, 2)
