from report_generator import generate_valhalla_premium_tax_report

sample_data = {
    "client_name": "Nathan Deratany",
    "tax_year": 2025,
    "filing_status": "Head of Household",
    "dependents": 2,
    "agi": 24266,
    "taxable_income": 0,
    "total_tax": 3429,
    "refund": 6731,
    "schedule_c_gross_revenue": 216265,
    "schedule_c_net_profit": 24266,
    "contract_labor": 107920,
    "depreciation": 16624,
    "vehicle_deduction": 4628,
    "qbi_deduction": 4510,
}

if __name__ == "__main__":
    output_file = generate_valhalla_premium_tax_report(
        sample_data,
        "valhalla_premium_report.pdf"
    )
    print(f"Generated: {output_file}")