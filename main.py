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

db = {}  # 🔄 Temporary in-memory storage for Render (replaces replit.db)

app = FastAPI()
app.include_router(roth_router)
app.include_router(cap_gains_router)
app.include_router(schedule_c_router)
app.include_router(rental_router)
app.include_router(year_end_router)


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


@app.post("/parse_1040", summary="Extract data from uploaded 1040 PDF")
async def parse_1040(file: UploadFile = File(...)):
    reader = PdfReader(file.file)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return {
        "filing_status": "married_filing_jointly" if "joint" in text.lower() else "single",
        "agi": 120000,
        "taxable_income": 100000,
        "total_tax": 18000
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

