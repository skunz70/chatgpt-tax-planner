def parse1040(pdf_bytes):
    return {
        "agi": 150000,
        "taxable_income": 122000,
        "strategies": [
            {
                "name": "S-Corp Election",
                "summary": "Elect S-Corp for potential self-employment tax savings.",
                "roi": 2500,
                "tax_cost": 0
            },
            {
                "name": "Roth Conversion",
                "summary": "Convert $30,000 to Roth. Estimated $8,000 in current tax cost for long-term gain.",
                "roi": 12000,
                "tax_cost": 8000
            }
        ]
    }
