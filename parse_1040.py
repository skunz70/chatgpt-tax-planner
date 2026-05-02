from PyPDF2 import PdfReader
import re

def parse1040(pdf_bytes):
    with open("temp_1040.pdf", "wb") as f:
        f.write(pdf_bytes)

    reader = PdfReader("temp_1040.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Extract using regex
    def extract(pattern):
        match = re.search(pattern, text)
        return match.group(1).replace(",", "").strip() if match else None

    agi = extract(r"Adjusted Gross Income.*?(\d[\d,]*)")
    taxable_income = extract(r"Taxable Income.*?(\d[\d,]*)")

    # Example strategy logic (you can make this dynamic later)
    strategies = []
    if agi and int(agi) > 100000:
        strategies.append({
            "name": "S-Corp Election",
            "summary": "Elect S-Corp for potential self-employment tax savings.",
            "roi": 2500,
            "tax_cost": 0
        })

    if taxable_income and int(taxable_income) > 80000:
        strategies.append({
            "name": "Roth Conversion",
            "summary": "Convert $30,000 to Roth. Estimated $8,000 in current tax cost for long-term gain.",
            "roi": 12000,
            "tax_cost": 8000
        })

    return {
        "agi": agi,
        "taxable_income": taxable_income,
        "strategies": strategies
    }

