from fastapi import FastAPI, status, HTTPException, Depends, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse
from schemas import UserOut, UserAuth, TokenSchema, SystemUser
from replit import db
from uuid import uuid4
from utils import get_hashed_password, create_access_token, create_refresh_token, verify_password
from deps import get_current_user
from PyPDF2 import PdfReader

app = FastAPI()

@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def docs():
    return RedirectResponse(url="/docs")

@app.post("/signup", response_model=UserOut)
async def signup(data: UserAuth):
    if db.get(data.email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")
    hashed = get_hashed_password(data.password)
    user = {"email": data.email, "password": hashed, "id": str(uuid4())}
    db[data.email] = user
    return UserOut(**user)

@app.post("/login", response_model=TokenSchema)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = db.get(form.username)
    if not user or not verify_password(form.password, user["password"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid credentials")
    return TokenSchema(access_token=create_access_token(user["email"]), refresh_token=create_refresh_token(user["email"]))

@app.get("/me", response_model=UserOut)
async def me(user: SystemUser = Depends(get_current_user)):
    return user

@app.post("/parse_1040")
async def parse_1040(file: UploadFile = File(...)):
    reader = PdfReader(file.file)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return {"agi": 120000, "filing_status": "single", "taxable_income": 100000}

@app.post("/project_tax")
async def project_tax(data: dict):
    agi = data.get("current_agi", 0)
    add_inc = data.get("additional_income", 0)
    contrib = data.get("retirement_contributions", 0)
    total = agi + add_inc
    tax = round(total * 0.22 - contrib, 2)
    return {"projected_agi": total, "projected_tax_liability": tax, "marginal_rate": "22%"}

@app.post("/recommend_strategies")
async def recommend_strategies(data: dict):
    agi = data.get("agi", 0)
    biz = data.get("business_income", 0)
    filing = data.get("filing_status", "")
    strategies = []
    if agi > 100000 and filing == "married_filing_jointly":
        strategies.append("Maximize traditional IRA contributions")
    if biz > 0:
        strategies.append("Evaluate S‑Corp election")
    return {"strategies": strategies}
