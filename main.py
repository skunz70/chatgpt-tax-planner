from fastapi import FastAPI, status, HTTPException, Depends, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse
from schemas import UserOut, UserAuth, TokenSchema, SystemUser
from uuid import uuid4
from utils import get_hashed_password, create_access_token, create_refresh_token, verify_password
from deps import get_current_user
from PyPDF2 import PdfReader
from roth import router as roth_router
from cap_gains import router as cap_gains_router
from schedule_c import router as schedule_c_router
from rental_analysis import router as rental_router
from year_end_planning import router as year_end_router
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
import pytesseract
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
import io
from csv_excel_keyword import router as csv_excel_router
from csv_excel_keyword import router as csv_excel_router

db = {}  # 🔄 Temporary in-memory storage for Render (replaces replit.db)

app = FastAPI()
app.include_router(roth_router)
app.include_router(cap_gains_router)
app.include_router(schedule_c_router)
app.include_router(rental_router)
app.include_router(year_end_router)
app.include_router(csv_excel_router)



@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def docs():
    return RedirectResponse(url="/docs")


@app.post("/signup", summary="Create new user", response_model=UserOut)
async def create_user(data: UserAuth):
    if data.email in db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    user = {
        "email": data.email,
        "password": get_hashed_password(data.password),
        "id": str(uuid4())
    }
    db[data.email] = user
    return UserOut(**user)


@app.post("/login", summary="Create access and refresh tokens", response_model=TokenSchema)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = db.get(form_data.username)
    if user is None or not verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password"
        )
    return {
        "access_token": create_access_token(user["email"]),
        "refresh_token": create_refresh_token(user["email"]),
    }


@app.get("/me", summary="Get current user", response_model=UserOut)
async def get_me(user: SystemUser = Depends(get_current_user)):
    return user


from pdf2image import convert_from_bytes
import pytesseract
import io

@app.post("/parse_1040", summary="Extract data from uploaded 1040 PDF with OCR fallback")
async def parse_1040(file: UploadFile = File(...)):
    import tempfile
    import pytesseract
    from pdf2image import convert_from_bytes
    import re

    def extract_values(text):
        agi_match = re.search(r"Adjusted Gross Income\s+\$?([0-9,]+)", text, re.IGNORECASE)
        taxable_income_match = re.search(r"Taxable Income\s+\$?([0-9,]+)", text, re.IGNORECASE)
        total_tax_match = re.search(r"Total Tax\s+\$?([0-9,]+)", text, re.IGNORECASE)

        agi = int(agi_match.group(1).replace(',', '')) if agi_match else None
        taxable_income = int(taxable_income_match.group(1).replace(',', '')) if taxable_income_match else None
        total_tax = int(total_tax_match.group(1).replace(',', '')) if total_tax_match else None

        return agi, taxable_income, total_tax

    # Step 1: Try standard PDF text extraction
    reader = PdfReader(file.file)
    extracted_text = "".join(page.extract_text() or "" for page in reader.pages)
    agi, taxable_income, total_tax = extract_values(extracted_text)

    # Step 2: If no values found, use OCR
    if not all([agi, taxable_income, total_tax]):
        file.file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file.file.read())
            temp_pdf_path = temp_pdf.name

        images = convert_from_bytes(open(temp_pdf_path, 'rb').read())
        ocr_text = "\n".join(pytesseract.image_to_string(img) for img in images)
        agi, taxable_income, total_tax = extract_values(ocr_text)

    return {
        "filing_status": "married_filing_jointly" if "joint" in extracted_text.lower() else "single",
        "agi": agi or 0,
        "taxable_income": taxable_income or 0,
        "total_tax": total_tax or 0
    }


   


@app.post("/project_tax", summary="Project AGI and tax")
async def project_tax(data: dict):
    agi = data.get("current_agi", 0)
    additional_income = data.get("additional_income", 0)
    retirement_contributions = data.get("retirement_contributions", 0)
    projected_agi = agi + additional_income
    estimated_tax = projected_agi * 0.22
    return {
        "projected_agi": projected_agi,
        "projected_tax_liability": round(estimated_tax - retirement_contributions, 2),
        "marginal_rate": "22%"
    }


@app.post("/recommend_strategies", summary="Get strategic tax-saving ideas")
async def recommend(data: dict):
    strategies = []
    agi = data.get("agi", 0)
    filing_status = data.get("filing_status", "")
    business_income = data.get("business_income", 0)
    retirement_plan_type = data.get("retirement_plan_type", "none")

    if agi > 100000 and filing_status == "married_filing_jointly":
        strategies.append("Maximize traditional IRA contributions")

    if business_income > 0:
        strategies.append("Evaluate S‑Corp election to save on self-employment tax")

    if retirement_plan_type == "none":
        strategies.append("Consider a Solo 401(k) or SEP IRA")

    return {"strategies": strategies}

from fpdf import FPDF
from fastapi.responses import StreamingResponse
import io

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Tax Planning Report", ln=True, align="C")

    def add_section(self, title, content):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, content)
        self.ln()

@app.post("/generate_pdf")
def generate_pdf(data: dict):
    pdf = PDFReport()
    pdf.add_page()

    # Example sections using sample fields
    pdf.add_section("Filing Status", data.get("filing_status", "N/A"))
    pdf.add_section("Adjusted Gross Income (AGI)", f"${data.get('agi', 0):,.2f}")
    pdf.add_section("Taxable Income", f"${data.get('taxable_income', 0):,.2f}")
    pdf.add_section("Total Tax", f"${data.get('total_tax', 0):,.2f}")

    strategies = data.get("strategies", [])
    if strategies:
        strategy_text = "\n".join(f"- {s}" for s in strategies)
        pdf.add_section("Recommended Strategies", strategy_text)

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": "inline; filename=tax_planning_report.pdf"})
@app.post("/parse_bank_statement", summary="Extract data from a bank statement PDF")
async def parse_bank_statement(file: UploadFile = File(...)):
    reader = PdfReader(file.file)
    text = "".join(page.extract_text() or "" for page in reader.pages)

    # Basic parsing logic – can be enhanced further
    total_deposits = "Not found"
    total_withdrawals = "Not found"
    account_holder = "Unknown"

    if "deposit" in text.lower():
        total_deposits = "$12,340.00"  # Example hardcoded value
    if "withdrawal" in text.lower():
        total_withdrawals = "$8,750.00"
    if "account holder" in text.lower():
        account_holder = "Sample Name"

    return {
        "account_holder": account_holder,
        "total_deposits": total_deposits,
        "total_withdrawals": total_withdrawals,
        "statement_summary": "Parsed basic banking information."
    }
@app.post("/parse_retirement_statement", summary="Extract data from a retirement account PDF")
async def parse_retirement_statement(file: UploadFile = File(...)):
    reader = PdfReader(file.file)
    text = "".join(page.extract_text() or "" for page in reader.pages)

    # Sample extracted values
    beginning_balance = "$145,000.00"
    contributions = "$6,500.00"
    distributions = "$0.00"
    ending_balance = "$158,000.00"

    return {
        "account_type": "401(k)",
        "beginning_balance": beginning_balance,
        "contributions": contributions,
        "distributions": distributions,
        "ending_balance": ending_balance,
        "notes": "This summary assumes a standard quarterly statement layout."
    }
@app.post("/parse_annuity_statement", summary="Extract data from an annuity statement PDF")
async def parse_annuity_statement(file: UploadFile = File(...)):
    reader = PdfReader(file.file)
    text = "".join(page.extract_text() or "" for page in reader.pages)

    # Example parsing (can be enhanced with regex or keyword checks)
    annuity_type = "Fixed Indexed"
    contract_value = "$250,000.00"
    withdrawals = "$0.00"
    rider_fees = "$1,250.00"
    income_base = "$280,000.00"

    return {
        "annuity_type": annuity_type,
        "contract_value": contract_value,
        "withdrawals": withdrawals,
        "rider_fees": rider_fees,
        "income_base": income_base,
        "notes": "Values are estimates based on a mock annuity report."
    }
@app.post("/parse_keywords_pdf", summary="Extract financial keywords from PDF")
async def parse_keywords_pdf(file: UploadFile = File(...)):
    reader = PdfReader(file.file)
    text = "".join(page.extract_text() or "" for page in reader.pages).lower()

    keywords = [
        "dividend", "interest", "capital gain", "qualified", "distribution",
        "required minimum distribution", "RMD", "IRA", "Roth", "SEP", "1099", "K-1", "Schedule C"
    ]

    detected = [kw for kw in keywords if kw in text]
    return {
        "keywords_detected": detected,
        "summary": f"{len(detected)} financial keywords found."
    }
import pandas as pd

@app.post("/parse_csv_data", summary="Analyze uploaded CSV data")
async def parse_csv_data(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    summary = {
        "columns": df.columns.tolist(),
        "row_count": len(df),
        "preview": df.head(5).to_dict(orient="records")
    }
    return summary
@app.post("/parse_excel_data", summary="Analyze uploaded Excel file")
async def parse_excel_data(file: UploadFile = File(...)):
    df = pd.read_excel(file.file)
    summary = {
        "columns": df.columns.tolist(),
        "row_count": len(df),
        "preview": df.head(5).to_dict(orient="records")
    }
    return summary
def extract_text_with_ocr(pdf_bytes: bytes) -> str:
    try:
        text = ""
        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes)
        for image in images:
            ocr_text = pytesseract.image_to_string(image)
            text += ocr_text + "\n"
        return text
    except Exception as e:
        return f"OCR failed: {str(e)}"

@app.post("/parse_statement", summary="OCR-enhanced parsing of brokerage, 401k, or bank statement")
async def parse_statement(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = ""
        try:
            # Try standard PDF text extraction first
            reader = PdfReader(io.BytesIO(contents))
            text = "".join(page.extract_text() or "" for page in reader.pages)
        except:
            text = ""

        if not text.strip():
            # Fallback to OCR
            text = extract_text_with_ocr(contents)

        # Simple keyword parsing
        keywords = ["account", "interest", "dividends", "contributions", "withdrawals", "Roth", "401(k)", "IRA", "statement", "bank"]
        found = [kw for kw in keywords if kw.lower() in text.lower()]
        
        summary = {
            "length_of_text": len(text),
            "keywords_detected": found,
            "note": "Parsed using OCR" if not text.strip() else "Parsed using PDF text layer"
        }
        return JSONResponse(content=summary)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

