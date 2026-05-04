import os
import io
from fpdf import FPDF
import tempfile
import inspect

from fastapi import FastAPI, Response, UploadFile, File, Depends, HTTPException, Request, Body, status
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi.security import OAuth2PasswordRequestForm

# ---- ONE FastAPI app instance ----
app = FastAPI()

# ---- CORS middleware ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Serve plugin manifest ----
import pathlib

@app.get("/.well-known/ai-plugin.json", response_class=Response, include_in_schema=False)
def serve_plugin_manifest():
    manifest_path = pathlib.Path(__file__).parent.joinpath(".well-known", "ai-plugin.json")
    if not manifest_path.is_file():
        return Response("Plugin manifest not found", media_type="text/plain", status_code=404)
    return Response(manifest_path.read_text(encoding="utf-8"), media_type="application/json")





# ---- Serve OpenAPI spec ----

@app.get("/openapi.yaml", response_class=Response, include_in_schema=False)
def serve_openapi_spec():
    spec_path = pathlib.Path(__file__).parent.joinpath("openapi.yaml")
    if not spec_path.is_file():
        return Response("OpenAPI spec not found", media_type="text/plain", status_code=404)
    return Response(spec_path.read_text(encoding="utf-8"), media_type="application/x-yaml")




# ---- Import your internal routes AFTER mounting ----
from year_end_planning import year_end_plan
from withdrawal_optimizer import router as withdrawal_optimizer_router
from auto_tax_plan import router as auto_tax_plan_router
from parse_1040 import parse1040
from report_generator import generate_tax_plan_pdf









   

    








# Include routers (order matters)
app.include_router(auto_tax_plan_router)
app.include_router(withdrawal_optimizer_router)


from multi_year_roth import compare_scenarios

from uuid import uuid4
import pytesseract
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
import io
from matplotlib import pyplot as plt
from fpdf import FPDF
import re

import re

def clean_text(text: str) -> str:
    """
    Strips emojis, bullets, smart quotes, and ensures the text is Latin-1 encodable.
    """
    if not text:
        return ""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove emojis and non-ASCII
    replacements = {
        "“": '"', "”": '"',
        "‘": "'", "’": "'",
        "•": "-", "–": "-", "—": "-"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


from typing import Optional

from schemas import UserOut, UserAuth, TokenSchema, SystemUser
from utils import get_hashed_password, create_access_token, create_refresh_token, verify_password
from deps import get_current_user

# Custom routers
from roth import router as roth_router
from cap_gains import router as cap_gains_router
from schedule_c import router as schedule_c_router
from rental_analysis import router as rental_router
from year_end_planning import router as year_end_router
from csv_excel_keyword import router as csv_excel_router
from multi_year_roth import router as multi_year_roth_router

# Tax logic and PDF generation
from arizona_tax import calculate_arizona_tax
from report_generator import generate_tax_plan_pdf, generate_smart_strategy_pdf


from pydantic import BaseModel
class ActionRequest(BaseModel):
    action: str
    income: float | None = None
    client_name: str | None = None
    tax_year: str | None = "2025"
    filing_status: str | None = None
    state: str | None = None
    agi: float | None = None
    taxable_income: float | None = None
    total_tax: float | None = None
    federal_withholding: float | None = None
    estimated_payments: float | None = None
    total_payments: float | None = None
    refund: float | None = None
    balance_due: float | None = None
    standard_deduction: float | None = None
    marginal_rate: float | None = None
    effective_rate: float | None = None
    w2_income: float | None = None
    interest_income: float | None = None
    dividend_income: float | None = None
    capital_gains: float | None = None
    ira_distributions: float | None = None
    pension_income: float | None = None
    business_income: float | None = None
    rental_income: float | None = None
    self_employment_tax: float | None = None
    retirement_contributions: float | None = None
    itemized_deductions: float | None = None
    mortgage_interest: float | None = None
    charitable_contributions: float | None = None
    credits: float | None = None
    confidence_engine: str | None = None
    planning_status: str | None = None
    additional_input: str | None = None

    model_config = {"extra": "allow"}

class YearEndPlanInput(BaseModel):
    filing_status: str
    w2_income: float = 0
    business_income: float = 0
    capital_gains: float = 0
    itemized_deductions: float = 0
    retirement_contributions: float = 0
    hsa_contributions: float = 0
    estimated_payments: float = 0





import re
from fpdf import FPDF
import tempfile
from fastapi.responses import FileResponse

def safe_text(value):
    try:
        text = str(value)
        # Strip emojis, bullets, and non-ASCII
        clean = re.sub(r"[^\x00-\x7F]+", "", text)
        return clean.encode("latin-1", errors="ignore").decode("latin-1")
    except Exception:
        return str(value)

   
       









# Then your routers:

# ... other routers



db = {}  # 🔄 Temporary in-memory storage for Render (replaces replit.db)



app.include_router(roth_router)
app.include_router(cap_gains_router)
app.include_router(schedule_c_router)
app.include_router(rental_router)
app.include_router(year_end_router)
app.include_router(csv_excel_router)


@app.post("/gpt-tax-router")
async def tax_router(request: ActionRequest):
    data = request.dict()

    action_map = {
        "tax_snapshot_summary": tax_snapshot_summary,
        "roth_conversion": roth_conversion,
        "multi_year_bracket": multi_year_bracket,
        "capital_gains_review": capital_gains_review,
        "withholding_review": withholding_review,
        "ira_hsa_review": ira_hsa_review,
        "deduction_bunching": deduction_bunching,
        "charitable_giving": charitable_giving,
        "business_tax_snapshot": business_tax_snapshot,
        "self_employed_optimizer": self_employed_optimizer,
        "social_security_planner": social_security_planner,
        "state_tax_strategy": state_tax_strategy,
        "aca_health_review": aca_health_review,
        "real_estate_passive": real_estate_passive,
        "bracket_analyzer": bracket_analyzer,
        "year_end_moves": year_end_moves,
        "client_specific": client_specific,
        "doc_risk_review": doc_risk_review,
        "dependent_credit_review": dependent_credit_review,
        "prompt_helper": prompt_helper,
        "quick_entry_plan": quick_entry_plan,
        "smart_strategy_report": generate_full_valhalla_pdf_report,
        "scenario_comparison": compare_scenarios,
        "generate_strategy_with_roi": year_end_plan,
    }

    if request.action not in action_map:
        raise HTTPException(status_code=400, detail="Invalid action specified.")

    action_result = action_map[request.action](data)
    if inspect.isawaitable(action_result):
        return await action_result
    return action_result
    


def _to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_strategy_priorities(data: dict):
    priorities = []

    balance_due = _to_float(data.get("balance_due"))
    taxable_income = _to_float(data.get("taxable_income"))
    retirement_contributions = _to_float(data.get("retirement_contributions"))
    agi = _to_float(data.get("agi"))
    business_income = _to_float(data.get("business_income"))
    rental_income = _to_float(data.get("rental_income"))
    capital_gains = _to_float(data.get("capital_gains"))

    if balance_due is not None and balance_due > 0:
        priorities.append({
            "strategy_name": "Withholding Optimization",
            "why_it_matters": "A current balance due indicates underwithholding or underpayment risk that can continue into next year.",
            "estimated_tax_impact": f"Potentially reduce next filing season balance due by up to ${balance_due:,.0f} through W-4 and estimate adjustments.",
            "difficulty_level": "Low",
            "recommended_timing": "Immediately; update withholding before the next payroll cycle.",
            "advisor_note": "Coordinate paycheck withholding and quarterly estimates to smooth cash flow and reduce penalty risk."
        })

    if taxable_income is not None:
        priorities.append({
            "strategy_name": "Bracket Management",
            "why_it_matters": "Taxable income determines marginal rate exposure and where proactive income and deduction timing can help.",
            "estimated_tax_impact": "Moderate; typically 1-3% of taxable income through timing and bracket-capacity planning.",
            "difficulty_level": "Medium",
            "recommended_timing": "During mid-year and year-end projection cycles.",
            "advisor_note": "Model income acceleration/deferral and deduction timing to avoid unnecessary marginal-rate creep."
        })

    low_retirement = retirement_contributions is None
    if not low_retirement and agi is not None:
        low_retirement = retirement_contributions < max(6000.0, agi * 0.05)

    if low_retirement:
        priorities.append({
            "strategy_name": "Retirement Contribution Optimization",
            "why_it_matters": "Tax-deferred or tax-free retirement contributions can lower current taxable income and improve long-term compounding.",
            "estimated_tax_impact": "Moderate to high depending on contribution room and marginal bracket.",
            "difficulty_level": "Low to Medium",
            "recommended_timing": "Increase deferrals now; finalize contribution limits before year-end deadlines.",
            "advisor_note": "Prioritize employer-plan deferrals and evaluate IRA/HSA eligibility for additional tax leverage."
        })

    if rental_income is not None and rental_income != 0:
        priorities.append({
            "strategy_name": "Rental Activity Optimization",
            "why_it_matters": "Rental activity can create deduction timing opportunities and passive-loss planning considerations.",
            "estimated_tax_impact": "Varies; depends on depreciation, repairs, and passive-loss utilization.",
            "difficulty_level": "Medium",
            "recommended_timing": "Before major property expenses and before year-end close.",
            "advisor_note": "Review Schedule E treatment, documentation, and depreciation strategy to maximize allowable deductions."
        })

    if business_income is not None and business_income > 0:
        priorities.append({
            "strategy_name": "Business Tax Optimization",
            "why_it_matters": "Business income may qualify for planning across deductions, entity structure, and retirement plan design.",
            "estimated_tax_impact": "Moderate to high, especially when QBI and deduction planning are available.",
            "difficulty_level": "Medium to High",
            "recommended_timing": "Quarterly, with a deeper review before year-end.",
            "advisor_note": "Evaluate QBI-sensitive planning, accountable-plan use, and retirement contributions tied to business cash flow."
        })

    if not priorities and capital_gains is not None and capital_gains > 0:
        priorities.append({
            "strategy_name": "Capital Gains Coordination",
            "why_it_matters": "Realized gains can increase current-year tax and interact with bracket thresholds.",
            "estimated_tax_impact": "Moderate; depends on gain size and holding period.",
            "difficulty_level": "Medium",
            "recommended_timing": "Before additional asset sales and at year-end.",
            "advisor_note": "Use gain/loss netting and holding-period review to improve after-tax results."
        })

    top_priorities = priorities[:3]
    for idx, item in enumerate(top_priorities, start=1):
        item["priority_rank"] = idx

    return top_priorities

async def generate_full_valhalla_pdf_report(data: dict):
    """
    Master report workflow.
    This prevents the GPT from returning a plain, cookie-cutter text report.
    """

    try:
        # 1. Run the strategy engine
        strategy_result = smart_strategy_report(data)
        if inspect.isawaitable(strategy_result):
            strategy_result = await strategy_result

        if isinstance(strategy_result, dict) and strategy_result.get("status") == "needs_parsed_1040":
            # Pass through structured validation feedback for GPT router clients.
            return strategy_result

        if isinstance(strategy_result, dict):
            strategy_text = (
                strategy_result.get("report_text")
                or strategy_result.get("report")
                or strategy_result.get("summary")
                or str(strategy_result)
            )
        else:
            strategy_text = str(strategy_result)

        client_name = data.get("client_name", "Client")
        tax_year = data.get("tax_year", "2024")

        agi = data.get("agi") or 0
        taxable_income = data.get("taxable_income") or 0
        total_tax = data.get("total_tax") or 0
        filing_status = data.get("filing_status") or "Unknown"
        refund = data.get("refund", None)
        balance_due = data.get("balance_due", None)
        marginal_rate = data.get("marginal_rate", "N/A")
        effective_rate = data.get("effective_rate")
        confidence_engine = data.get("confidence_engine", {}) or {}
        confidence_score = confidence_engine.get("confidence_score", data.get("confidence_score", "N/A"))
        missing_fields = confidence_engine.get("missing_fields", [])
        planning_status = data.get("planning_status", "unknown")
        core_fields = ["agi", "taxable_income", "total_tax", "filing_status"]
        missing_core_fields = [field for field in core_fields if not data.get(field)]
        combined_missing_fields = list(dict.fromkeys([*missing_fields, *missing_core_fields]))

        def _to_number(value):
            try:
                if value in (None, "", "N/A"):
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        def _money(value):
            value_num = _to_number(value)
            return f"${value_num:,.0f}" if value_num is not None else "N/A"

        def _percent(value):
            value_num = _to_number(value)
            return f"{value_num:.0f}%" if value_num is not None else "N/A"

        agi_num = _to_number(agi)
        taxable_num = _to_number(taxable_income)
        total_tax_num = _to_number(total_tax)
        effective_rate_num = _to_number(effective_rate)
        if effective_rate_num is None and agi_num and agi_num > 0 and total_tax_num is not None:
            effective_rate_num = round((total_tax_num / agi_num) * 100, 2)
        effective_rate_display = _percent(effective_rate_num)
        marginal_rate_display = _percent(str(marginal_rate).replace("%", "")) if marginal_rate not in (None, "N/A") else "N/A"
        marginal_rate_decimal = (_to_number(str(marginal_rate).replace("%", "")) or 0) / 100

        # 2. Force polished Valhalla structure
        full_report_text = f"""
Valhalla Tax Services
Tax Planning Report

Client: {client_name}
Tax Year: {tax_year}

EXECUTIVE SUMMARY
This section provides a high-level overview of the client's tax profile and planning focus.

The client has adjusted gross income of {_money(agi)}, taxable income of {_money(taxable_income)}, and total federal tax of {_money(total_tax)}. The primary focus is to reduce avoidable tax drag, improve withholding accuracy, and coordinate federal and Arizona planning opportunities.

CONFIRMED TAX DATA SUMMARY
This section lists the validated tax inputs used to prepare this planning report.

Filing Status: {filing_status}
State: {data.get("state", "Arizona")}
Adjusted Gross Income: {_money(agi)}
Taxable Income: {_money(taxable_income)}
Total Federal Tax: {_money(total_tax)}
Federal Withholding: {_money(data.get("federal_withholding", "N/A"))}
Refund: {_money(refund)}
Balance Due: {_money(balance_due)}
Marginal Rate: {marginal_rate_display}
Effective Rate: {effective_rate_display}

CURRENT TAX POSITION
This section explains the current federal tax posture and cash-flow implications.

With AGI of {_money(agi)} and taxable income of {_money(taxable_income)}, the current effective federal tax burden is {effective_rate_display}. The marginal rate indicator is {marginal_rate_display}. This supports estimated payment calibration and year-end optimization decisions.

TAX BRACKET ANALYSIS
This section shows the core tax-rate math used to frame strategy decisions.

Current effective rate calculation: Total Tax ÷ AGI = {_money(total_tax)} ÷ {_money(agi)} = {effective_rate_display}. This baseline helps compare the cost of additional income versus tax savings from deductions, deferrals, and credits.

STRATEGIC TAX PLAN
This section prioritizes practical tax strategies with concise calculations and implementation context.

1. **Retirement Contribution Optimization**
Reported retirement contributions are {_money(data.get("retirement_contributions", "N/A"))}. At a marginal rate of {marginal_rate_display}, each additional $1,000 pre-tax contribution may reduce federal income tax by approximately {_money(1000 * marginal_rate_decimal)}. Retirement contributions generally reduce income tax, not self-employment tax.

2. **Roth Conversion Strategy**
Use current taxable income of {_money(taxable_income)} to evaluate bracket capacity for partial Roth conversions before year-end bracket compression.

3. **Withholding Correction Strategy**
Current withholding is {_money(data.get("federal_withholding", "N/A"))} against total tax of {_money(total_tax)}. Net position check: {_money((_to_number(data.get("federal_withholding")) or 0) - (total_tax_num or 0))}. If negative, increase W-4 withholding or estimated payments.

4. **Deduction Timing Strategy**
Standard deduction: {_money(data.get("standard_deduction", "N/A"))}; Itemized deductions: {_money(data.get("itemized_deductions", "N/A"))}; Mortgage interest: {_money(data.get("mortgage_interest", "N/A"))}; Charitable contributions: {_money(data.get("charitable_contributions", "N/A"))}. Timing deductions into one year can increase marginal deduction value. For Schedule C deductions, rough planning math can include income-tax savings (deduction x marginal rate) plus self-employment tax savings (deduction x 15.3%).

5. **Income and Benefit Coordination**
Business income: {_money(data.get("business_income", "N/A"))}; Rental income: {_money(data.get("rental_income", "N/A"))}; Capital gains: {_money(data.get("capital_gains", "N/A"))}. Coordinate timing with retirement contributions and withholding updates.

6. **Estimated Tax and Cash Flow Planning**
Refund reported: {_money(refund)}; Balance due: {_money(balance_due)}. Quarterly catch-up example: {_money((_to_number(balance_due) or 0) / 4)}.

7. **Arizona State Strategy**
State listed: {data.get("state", "Arizona")}. Align federal moves with Arizona treatment of deductions, retirement contributions, and payment schedules.

{strategy_text}

ACTION PLAN TIMELINE
This section defines execution sequencing across immediate, mid-year, and year-end windows.

Immediate (next 30 days): Validate missing fields, confirm withholding-to-liability alignment, and prioritize top projected after-tax actions.
Mid-Year (next 3-6 months): Implement income-timing and deduction strategies while tracking estimated payments versus projected total tax.
Year-End (Q4 execution): Finalize bracket actions, complete contribution-based strategies, and run a pre-filing projection.

FINAL RECOMMENDATION
This section summarizes the client-ready execution focus.

Focus first on the highest-value planning items supported by return data. Prioritize strategies that reduce avoidable tax, improve long-term efficiency, and correct withholding issues before the next filing season.
"""

        full_report_text = (
            full_report_text
            .replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("–", "-")
            .replace("—", "-")
        )

        pdf_available = False
        pdf_message = "PDF generation failed, but your report is available above."
        try:
            pdf_result = await generate_strategy_pdf({
                "report_text": full_report_text,
                "client_name": client_name,
                "tax_year": tax_year,
            })
            if pdf_result is not None:
                pdf_available = True
                pdf_message = "Your PDF report is ready."
        except Exception:
            pdf_available = False
            pdf_message = "PDF generation failed, but your report is available above."

        return {
            "status": "success",
            "report_type": "valhalla_comprehensive_tax_plan",
            "report_text": full_report_text,
            "pdf_available": pdf_available,
            "pdf_message": pdf_message,
            "client_name": client_name,
            "tax_year": tax_year,
            "strategy_priorities": _build_strategy_priorities(data),
            "missing_fields": combined_missing_fields,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Report generation failed",
            "debug": str(e)
        }


# === Logic for Each Action ===

def tax_snapshot_summary(req): return {"summary": f"Tax summary for {req.get('tax_year') or 'current year'}"}

def roth_conversion(req): return {"conversion": f"Roth analysis for income {req.get('income') or 'N/A'}"}

def multi_year_bracket(req): return {"multi_year": "Multi-year bracket forecast"}

def capital_gains_review(req): return {"gains": "Capital gains strategy"}

def withholding_review(req): return {"withholding": "Check W-4 or estimated payments"}

def ira_hsa_review(req): return {"ira_hsa": "IRA and HSA review"}

def deduction_bunching(req): return {"bunching": "Itemized vs standard deduction analysis"}

def charitable_giving(req): return {"charity": "DAF and appreciated stock strategies"}

def business_tax_snapshot(req): return {"business": "QBI and entity structure analysis"}

def self_employed_optimizer(req): return {"self_employed": "SEP/Solo 401(k) and deductions"}

def social_security_planner(req): return {"ss": "Claiming strategy and taxability"}

def state_tax_strategy(req): return {"state": "Part-year, residency, and credits"}

def aca_health_review(req): return {"aca": "PTC, health insurance deduction"}

def real_estate_passive(req): return {"real_estate": "Passive losses, RE pro status"}

def bracket_analyzer(req): return {"bracket": "Marginal/effective tax rate"}

def year_end_moves(req): return {"moves": "Year-end tax strategy checklist"}

def client_specific(req): return {"client": "Customized plan based on profile"}

def doc_risk_review(req): return {"risk": "Audit, estate, document checklist"}

def dependent_credit_review(req): return {"credits": "CTC/ACTC multi-year eligibility"}

def prompt_helper(req): return {"prompts": "Reusable prompt guidance"}

    

# === Logic for Each Action ===

def tax_snapshot_summary(req): return {"summary": f"Tax summary for {req.get('tax_year') or 'current year'}"}

def roth_conversion(req): return {"conversion": f"Roth analysis for income {req.get('income') or 'N/A'}"}

def multi_year_bracket(req): return {"multi_year": "Multi-year bracket forecast"}

def capital_gains_review(req): return {"gains": "Capital gains strategy"}

def withholding_review(req): return {"withholding": "Check W-4 or estimated payments"}

def ira_hsa_review(req): return {"ira_hsa": "IRA and HSA review"}

def deduction_bunching(req): return {"bunching": "Itemized vs standard deduction analysis"}

def charitable_giving(req): return {"charity": "DAF and appreciated stock strategies"}

def business_tax_snapshot(req): return {"business": "QBI and entity structure analysis"}

def self_employed_optimizer(req): return {"self_employed": "SEP/Solo 401(k) and deductions"}

def social_security_planner(req): return {"ss": "Claiming strategy and taxability"}

def state_tax_strategy(req): return {"state": "Part-year, residency, and credits"}

def aca_health_review(req): return {"aca": "PTC, health insurance deduction"}

def real_estate_passive(req): return {"real_estate": "Passive losses, RE pro status"}

def bracket_analyzer(req): return {"bracket": "Marginal/effective tax rate"}

def year_end_moves(req): return {"moves": "Year-end tax strategy checklist"}

def client_specific(req): return {"client": "Customized plan based on profile"}

def doc_risk_review(req): return {"risk": "Audit, estate, document checklist"}

def dependent_credit_review(req): return {"credits": "CTC/ACTC multi-year eligibility"}

def prompt_helper(req): return {"prompts": "Reusable prompt guidance"}



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


from PyPDF2 import PdfReader
import io
import re
from pdf2image import convert_from_bytes
import pytesseract



def extract_1040_lines_from_text(text: str) -> tuple[dict, dict]:
    clean_text = text.replace("\r", "\n")
    normalized_lines = [line.strip() for line in clean_text.split("\n") if line.strip()]

    def _parse_amount(raw_amount: str):
        cleaned = re.sub(r"[^\d\-]", "", raw_amount or "")
        if cleaned in {"", "-"}:
            return None
        try:
            return int(cleaned)
        except ValueError:
            return None

    def _find_line_amount(line_number_pattern: str, line_label_patterns: list[str]):
        number_regex = r"\(?-?\$?\s*([0-9][0-9,]*)\)?"
        candidates = []
        for idx, line in enumerate(normalized_lines):
            line_ok = re.search(line_number_pattern, line, flags=re.IGNORECASE)
            label_ok = any(re.search(lp, line, flags=re.IGNORECASE) for lp in line_label_patterns)
            if not line_ok or not label_ok:
                continue
            found = re.findall(number_regex, line)
            if found:
                value = _parse_amount(found[-1])
                if value is not None:
                    candidates.append((value, idx, line))

        if not candidates:
            return None, None
        value, idx, matched_line = candidates[-1]
        excerpt = " ".join(normalized_lines[max(0, idx - 1): min(len(normalized_lines), idx + 2)])
        return value, excerpt

    field_rules = {
        "w2_wages_line_1a": (r"\b1a\b|\b1\s*a\b", [r"wages", r"w-?2"]),
        "additional_income_line_8": (r"\b8\b", [r"additional income", r"schedule\s*1"]),
        "total_income_line_9": (r"\b9\b", [r"total income"]),
        "agi": (r"\b11\b", [r"adjusted gross income", r"\bagi\b"]),
        "taxable_income": (r"\b15\b", [r"taxable income"]),
        "line_16_tax": (r"\b16\b", [r"\btax\b"]),
        "total_tax": (r"\b24\b", [r"total tax"]),
        "withholding": (r"\b25d\b|\b25\s*d\b", [r"withheld", r"withholding"]),
        "total_payments": (r"\b33\b", [r"total payments"]),
        "refund_line_34": (r"\b34\b", [r"refund"]),
        "balance_due": (r"\b37\b", [r"amount you owe", r"amount owed"]),
    }

    extracted = {}
    extraction_debug = {}
    for field, (line_pattern, labels) in field_rules.items():
        amount, excerpt = _find_line_amount(line_pattern, labels)
        extracted[field] = amount
        extraction_debug[field] = {
            "matched_excerpt": excerpt,
            "value": amount
        }

    extracted["estimated_payments"] = 0
    extraction_debug["estimated_payments"] = {
        "matched_excerpt": "Defaulted to 0. Form 1040 line 33 already includes withholding + all payment credits.",
        "value": 0
    }
    return extracted, extraction_debug

def ocr_extract_text(pdf_bytes: bytes) -> str:
    text = ""
    images = convert_from_bytes(pdf_bytes)
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return text

@app.post("/parse_1040", summary="Extract data from uploaded 1040 PDF with OCR fallback")
async def parse_1040(request: Request, body: dict = Body(default=None)):
    print("====== /parse_1040 HIT ======", flush=True)
    print("CONTENT TYPE:", request.headers.get("content-type"), flush=True)

    content_type = request.headers.get("content-type")
    pdf_bytes = None
    received_filename = None

    try:
        form = await request.form()
        for value in form.values():
            if hasattr(value, "filename") and hasattr(value, "read"):
                received_filename = value.filename
                pdf_bytes = await value.read()
                print("FILE FOUND VIA FORM", flush=True)
                break
    except Exception as e:
        print("FORM PARSE FAILED:", str(e), flush=True)

    if pdf_bytes is None and body:
        print("BODY RECEIVED:", body, flush=True)

        if "file_base64" in body:
            import base64
            pdf_bytes = base64.b64decode(body["file_base64"])
            print("FILE FOUND VIA BASE64", flush=True)

    if not pdf_bytes:
        return {
            "error": "No file detected",
            "debug": {
                "content_type": content_type,
                "body": body
            }
        }

    # keep the rest of your existing route below this line

    # Try OCR first
    text = ocr_extract_text(pdf_bytes)

    # Fallback: try reading embedded PDF text
    if not text or not text.strip():
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages_text)
        except Exception as e:
            return {
                "error": "PDF text extraction failed.",
                "detail": str(e),
                "received_filename": received_filename,
                "content_type": content_type
            }

    lines, extraction_debug = extract_1040_lines_from_text(text)

    validation_warnings = []

    agi = lines.get("agi")
    taxable_income = lines.get("taxable_income")
    total_tax = lines.get("total_tax")
    withholding = lines.get("withholding")
    total_payments = lines.get("total_payments")
    balance_due = lines.get("balance_due")
    total_income = lines.get("total_income_line_9")
    additional_income = lines.get("additional_income_line_8")
    w2_income = lines.get("w2_wages_line_1a")
    refund = lines.get("refund_line_34")
    line_16_tax = lines.get("line_16_tax")

    if agi is None:
        validation_warnings.append("AGI could not be confidently detected from Form 1040 Line 11.")

    if taxable_income is None:
        validation_warnings.append("Taxable income could not be confidently detected from Form 1040 Line 15.")

    if total_tax is None:
        validation_warnings.append("Total tax could not be confidently detected from Form 1040 Line 24.")

    if agi is not None and taxable_income is not None and taxable_income > agi:
        validation_warnings.append("Taxable income appears higher than AGI. Verify OCR extraction.")

    if agi is not None and total_tax is not None and total_tax > agi:
        validation_warnings.append("Total tax appears unusually high compared to AGI. Verify OCR extraction.")
    if agi is not None and agi < 0:
        validation_warnings.append("AGI is negative. Verify Form 1040 line 11 extraction before planning.")
    if agi is not None and agi > 0 and taxable_income == 0:
        validation_warnings.append("Taxable income is zero while AGI is positive. Verify Form 1040 line 15.")
    if (agi or 0) > 0 and (total_income or 0) > 0 and total_tax == 0:
        validation_warnings.append("Total tax is zero while income exists. Verify Form 1040 lines 16 and 24.")

    missing_fields = []
    for field_name in ["agi", "taxable_income", "total_tax", "withholding", "estimated_payments", "total_payments"]:
        if lines.get(field_name) is None:
            missing_fields.append(field_name)

    reconciliation_warnings = []
    reconciliation_checks = {
        "taxable_income_greater_than_agi": {
            "applicable": agi is not None and taxable_income is not None,
            "passed": not (agi is not None and taxable_income is not None and taxable_income > agi),
            "detail": "Taxable income should generally not exceed AGI."
        },
        "total_tax_greater_than_agi": {
            "applicable": agi is not None and total_tax is not None,
            "passed": not (agi is not None and total_tax is not None and total_tax > agi),
            "detail": "Total tax should generally not exceed AGI."
        },
        "payments_match_total": {
            "applicable": withholding is not None and lines.get("estimated_payments") is not None and total_payments is not None,
            "passed": None,
            "detail": "Withholding + estimated payments should roughly match total payments."
        },
        "balance_due_refund_logic": {
            "applicable": total_tax is not None and total_payments is not None and balance_due is not None,
            "passed": None,
            "detail": "If total payments are less than total tax, balance due should be positive."
        }
    }

    estimated_payments = lines.get("estimated_payments")

    if reconciliation_checks["payments_match_total"]["applicable"]:
        combined_payments = withholding + estimated_payments
        diff = abs(combined_payments - total_payments)
        payments_ok = diff <= 5
        reconciliation_checks["payments_match_total"]["passed"] = payments_ok
        reconciliation_checks["payments_match_total"]["difference"] = diff
        if not payments_ok:
            reconciliation_warnings.append(
                "Withholding plus estimated payments does not match total payments. Verify Form 1040 lines 25 and 33."
            )

    if reconciliation_checks["balance_due_refund_logic"]["applicable"]:
        expected_balance_due = total_tax - total_payments
        balance_ok = (expected_balance_due <= 0 and balance_due == 0) or (expected_balance_due > 0 and balance_due > 0)
        reconciliation_checks["balance_due_refund_logic"]["passed"] = balance_ok
        reconciliation_checks["balance_due_refund_logic"]["expected_balance_due"] = expected_balance_due
        if not balance_ok:
            reconciliation_warnings.append(
                "Balance due/refund logic appears inconsistent with total tax versus total payments."
            )

    confidence_score = 100
    confidence_score -= len(missing_fields) * 7
    confidence_score -= (len(validation_warnings) + len(reconciliation_warnings)) * 5
    if agi is not None and agi < 0:
        confidence_score -= 18
    if agi is not None and agi > 0 and taxable_income == 0:
        confidence_score -= 14
    if (agi or 0) > 0 and (total_income or 0) > 0 and total_tax == 0:
        confidence_score -= 16
    if taxable_income is not None and total_income is not None and taxable_income > total_income:
        confidence_score -= 10
        validation_warnings.append("Taxable income exceeds total income. Verify line mapping and OCR quality.")
    confidence_score = max(0, min(100, confidence_score))

    safe_to_plan = (
        confidence_score >= 75
        and agi is not None
        and taxable_income is not None
        and total_tax is not None
    )

    next_best_action = (
        "Safe to generate planning report"
        if safe_to_plan
        else "Review missing Form 1040 fields before generating planning report"
    )

    if confidence_score >= 85 and safe_to_plan:
        planning_status = "ready_for_planning"
        planning_recommendation = "Data is clean. Proceed with full tax planning."
    elif 60 <= confidence_score <= 84:
        planning_status = "needs_review"
        planning_recommendation = "Some key fields are missing. Review before planning."
    else:
        planning_status = "insufficient_data"
        planning_recommendation = "Insufficient data for reliable planning. Upload a clearer return."

    planner_input = StrategyROIInput(
        filing_status="single",
        w2_income=float(agi or 0),
        business_income=0,
        capital_gains=0,
        dividend_income=0,
        retirement_contributions=0,
        itemized_deductions=0,
        estimated_payments=float(total_payments or 0),
        state="AZ",
        show_pdf=False,
        strategy_flags=[
            "roth_conversion",
            "s_corp_election",
            "aca_optimization"
        ]
    )

    planner_result = None
    planner_error = None
    if safe_to_plan:
        try:
            planner_result = generate_strategy_with_roi(planner_input)
        except Exception as e:
            planner_error = "Tax planning report generation failed. Extracted tax data is still available."
            print(f"PLANNER GENERATION FAILED: {str(e)}", flush=True)
    else:
        planner_error = "Planning not generated because extraction confidence is below safe threshold."

    return {
        "status": "success",
        "received_filename": received_filename,
        "content_type": content_type,
        "filing_status": "unknown",
        "w2_income": w2_income,
        "additional_income": additional_income,
        "total_income": total_income,
        "agi": agi,
        "taxable_income": taxable_income,
        "line_16_tax": line_16_tax,
        "total_tax": total_tax,
        "withholding": withholding,
        "estimated_payments": estimated_payments,
        "total_payments": total_payments,
        "refund": refund,
        "balance_due": balance_due,
        "extraction_debug": extraction_debug,
        "validation_warnings": validation_warnings,
        "planning_status": planning_status,
        "planning_recommendation": planning_recommendation,
        "confidence_engine": {
            "extracted_lines": {
                "agi": agi,
                "taxable_income": taxable_income,
                "total_tax": total_tax,
                "withholding": withholding,
                "estimated_payments": estimated_payments,
                "total_payments": total_payments,
                "balance_due": balance_due
            },
            "missing_fields": missing_fields,
            "validation_warnings": validation_warnings + reconciliation_warnings,
            "reconciliation_checks": reconciliation_checks,
            "confidence_score": confidence_score,
            "safe_to_plan": safe_to_plan,
            "next_best_action": next_best_action
        },
        "planner_result": planner_result,
        "planner_error": planner_error
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

    agi = float(data.get("agi", 0) or 0)
    taxable_income = float(data.get("taxable_income", 0) or 0)
    filing_status = str(data.get("filing_status", "single")).lower()
    business_income = float(data.get("business_income", 0) or 0)
    capital_gains = float(data.get("capital_gains", 0) or 0)
    retirement_plan_type = str(data.get("retirement_plan_type", "none")).lower()

    if taxable_income == 0 and agi > 0:
        taxable_income = max(0, agi - 15000)

        # ---- Bracket Logic (Single + MFJ) ----
    if filing_status == "single" and taxable_income < 103350:
        room = 103350 - taxable_income
        strategies.append(
            f"Bracket management opportunity: taxable income is approximately ${taxable_income:,.0f}, leaving about ${room:,.0f} of room in the 22% federal bracket before reaching the 24% bracket."
        )

    elif filing_status in ["married_filing_jointly", "mfj"] and taxable_income < 206700:
        room = 206700 - taxable_income
        strategies.append(
            f"Bracket management opportunity: taxable income is approximately ${taxable_income:,.0f}, leaving about ${room:,.0f} of room in the 22% federal bracket before reaching the 24% bracket."
        )

    # ---- NIIT Warning ----
    if (filing_status == "single" and agi > 180000) or (
        filing_status in ["married_filing_jointly", "mfj"] and agi > 230000
    ):
        strategies.append(
            "NIIT warning: income is approaching or exceeding Net Investment Income Tax thresholds. Review capital gains, dividends, and passive income exposure."
        )

    # ---- Roth Phaseout Awareness ----
    if filing_status == "single" and agi > 150000:
        strategies.append(
            "Roth IRA eligibility may be limited due to income phaseout. Consider backdoor Roth strategies if applicable."
        )

    elif filing_status in ["married_filing_jointly", "mfj"] and agi > 236000:
        strategies.append(
            "Roth IRA eligibility may be limited for MFJ due to income phaseouts. Review backdoor Roth contribution options."
        )
        
   

   
  
    if retirement_plan_type == "none":
        strategies.append(
            "Retirement planning opportunity: review Traditional IRA, Roth IRA, 401(k), SEP IRA, or Solo 401(k) options to reduce current taxes or improve long-term tax-free income."
        )

    if business_income > 0:
        strategies.append(
            "Business planning opportunity: review QBI deduction eligibility, deductible business expenses, retirement plan options, and whether S-Corp analysis is appropriate."
        )

    if business_income >= 30000:
        strategies.append(
            "Entity review trigger: business income may be high enough to evaluate an S-Corp election after considering reasonable compensation, payroll costs, and compliance requirements."
        )

    if capital_gains > 0:
        strategies.append(
            "Capital gains planning opportunity: review whether gains can be harvested within the current tax bracket or offset with tax-loss harvesting."
        )
    else:
        strategies.append(
            "Investment tax review: if the client has a taxable brokerage account, review unrealized gains and losses before year-end."
        )

    strategies.append(
        "Withholding review: compare projected tax against federal withholding and estimated payments to avoid a balance due or underpayment penalties."
    )

    strategies.append(
        "Deduction review: compare the standard deduction against itemized deductions and consider bunching charitable gifts or deductible expenses if appropriate."
    )

    return {"strategies": strategies}

from fpdf import FPDF
from fastapi.responses import StreamingResponse
import io

class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Tax Planning Report", ln=True, align="C")

    def add_section(self, title, content):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.set_font("Helvetica", size=12)
        self.multi_cell(0, 10, content)
        self.ln()

from fastapi.responses import Response
from report_generator import generate_tax_plan_pdf

@app.post("/generate_pdf")
def generate_pdf(payload: dict):
    try:
        pdf_bytes = generate_tax_plan_pdf(
    data=payload
)
        return Response(content=pdf_bytes, media_type="application/pdf")
    except Exception as e:
        return {"error": f"PDF generation failed: {str(e)}"}

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
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import io

@app.post("/visualize_roth_conversion", summary="Visual Roth conversion impact")
async def visualize_roth_conversion(data: dict):
    current_agi = data.get("current_agi", 0)
    conversion_amount = data.get("conversion_amount", 0)
    new_agi = current_agi + conversion_amount

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["AGI Before", "AGI After"], [current_agi, new_agi])
    ax.set_title("Roth Conversion Impact on AGI")
    ax.set_ylabel("Income ($)")
    ax.bar_label(bars, fmt="%.0f")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
@app.post("/phaseout_checker", summary="Detect phaseouts for deductions and credits")
async def phaseout_checker(data: dict):
    income = data.get("income", 0)
    filing_status = data.get("filing_status", "single").lower()

    # 2025 phaseout ranges (simplified examples — update with full brackets as needed)
    phaseouts = [
        {
            "name": "Child Tax Credit",
            "start": 200000 if filing_status == "single" else 400000,
            "end": 240000 if filing_status == "single" else 440000,
        },
        {
            "name": "IRA Deduction (active participant)",
            "start": 77000 if filing_status == "single" else 123000,
            "end": 87000 if filing_status == "single" else 143000,
        },
        {
            "name": "Roth IRA Contribution",
            "start": 146000 if filing_status == "single" else 230000,
            "end": 161000 if filing_status == "single" else 240000,
        },
        {
            "name": "Student Loan Interest Deduction",
            "start": 75000 if filing_status == "single" else 155000,
            "end": 90000 if filing_status == "single" else 185000,
        },
    ]

    results = []

    for item in phaseouts:
        if income >= item["start"] and income <= item["end"]:
            results.append({
                "item": item["name"],
                "status": "⚠️ In Phaseout Range",
                "detail": f"Phaseout begins at ${item['start']:,} and ends at ${item['end']:,}."
            })
        elif income > item["end"]:
            results.append({
                "item": item["name"],
                "status": "❌ Fully Phased Out",
                "detail": f"Income exceeds ${item['end']:,}, item fully disallowed."
            })
        else:
            results.append({
                "item": item["name"],
                "status": "✅ Fully Allowed",
                "detail": f"Income below ${item['start']:,}, full benefit available."
            })

    return {
        "income": income,
        "filing_status": filing_status,
        "phaseout_results": results
    }
@app.post("/threshold_modeling", summary="Evaluate IRMAA, ACA, and NIIT thresholds")
async def threshold_modeling(data: dict):
    filing_status = data.get("filing_status", "married_filing_jointly").lower()
    agi = data.get("agi", 0)
    magi = data.get("magi", agi)  # fallback if MAGI isn't separately provided

    response = {"warnings": [], "threshold_results": {}}

    # --- NIIT Threshold ---
    niit_thresholds = {
        "single": 200000,
        "married_filing_jointly": 250000,
        "head_of_household": 200000,
        "married_filing_separately": 125000
    }
    niit_base = niit_thresholds.get(filing_status, 250000)
    if magi > niit_base:
        response["warnings"].append("⚠️ Subject to Net Investment Income Tax (NIIT) of 3.8%")
        response["threshold_results"]["niit_excess"] = magi - niit_base

    # --- IRMAA (2025 Part B premiums based on 2023 MAGI) ---
    irmaa_tiers = [
        (194000, 0), (246000, 1), (306000, 2), (366000, 3), (750000, 4)
    ] if filing_status == "married_filing_jointly" else [
        (97000, 0), (123000, 1), (153000, 2), (183000, 3), (500000, 4)
    ]
    irmaa_labels = [
        "Base Premium", "IRMAA Tier 1", "IRMAA Tier 2", "IRMAA Tier 3", "IRMAA Tier 4", "IRMAA Tier 5"
    ]
    tier = next((i for i, (limit, _) in enumerate(irmaa_tiers) if magi <= limit), 5)
    response["threshold_results"]["irmaa_tier"] = irmaa_labels[tier]

    # --- ACA Subsidy Eligibility (FPL guidelines simplified) ---
    aca_fpl_cutoff = 180000 if filing_status == "married_filing_jointly" else 90000
    if magi > aca_fpl_cutoff:
        response["warnings"].append("⚠️ May not qualify for ACA premium subsidies")
    else:
        response["warnings"].append("✅ Likely eligible for ACA premium subsidies")

    return response
@app.post("/compare_scenarios", summary="Compare two tax planning scenarios")
async def compare_scenarios(data: dict):
    baseline = data.get("baseline", {})
    alternative = data.get("alternative", {})
    
    def compute_projection(inputs):
        agi = inputs.get("agi", 0) + inputs.get("additional_income", 0)
        taxable_income = agi - inputs.get("deductions", 0)
        est_tax = taxable_income * 0.22  # Example marginal rate
        return {
            "projected_agi": agi,
            "taxable_income": taxable_income,
            "estimated_tax": round(est_tax, 2),
        }

    result = {
        "baseline": compute_projection(baseline),
        "alternative": compute_projection(alternative),
        "difference": {
            "agi_diff": compute_projection(alternative)["projected_agi"] - compute_projection(baseline)["projected_agi"],
            "tax_diff": compute_projection(alternative)["estimated_tax"] - compute_projection(baseline)["estimated_tax"],
        }
    }
    return result

from matplotlib import pyplot as plt
from fastapi.responses import StreamingResponse

@app.post("/generate_strategy_pdf")
async def generate_strategy_pdf(data: dict):
    report_text = data.get("report_text", "No report content provided.")
    client_name = data.get("client_name", "Client")
    tax_year = data.get("tax_year", "Tax Year")

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # --- LOGO (top right) ---
    logo_path = os.path.join(os.path.dirname(__file__), "valhalla_logo.jpg")
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=240, y=10, w=40)

    # --- HEADER TEXT ---
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Valhalla Tax Services", ln=True)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Tax Planning Report", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Client: {client_name}", ln=True)
    pdf.cell(0, 7, f"Tax Year: {tax_year}", ln=True)
    pdf.ln(5)

    # --- EXECUTIVE SUMMARY ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Executive Summary", ln=True)

    pdf.set_font("Arial", "", 10)

    summary_text = (
        f"This report outlines key tax planning opportunities for {client_name}. "
        f"Based on current income levels and tax position, there are opportunities "
        f"to reduce tax liability and improve long-term tax efficiency."
    )

    pdf.multi_cell(0, 6, summary_text)
    pdf.ln(5)
    
    for line in report_text.split("\n"):
        clean_line = line.strip()

        if "Valhalla Tax Services" in clean_line or "Tax Planning Report" in clean_line:
            continue

        if not clean_line:
            pdf.ln(3)
            continue

        # Bold section headers
        if (
            clean_line.isupper()
            or "Summary" in clean_line
            or "Position" in clean_line
            or "Analysis" in clean_line
            or "Steps" in clean_line
            or "Recommendation" in clean_line
        ):
            pdf.set_font("Arial", "B", 13)
            pdf.ln(2)
            pdf.multi_cell(0, 7, clean_line)
            pdf.ln(1)
            pdf.set_font("Arial", "", 10)
        else:
            pdf.multi_cell(0, 6, clean_line)
            pdf.ln(1)

    temp_dir = tempfile.gettempdir()
    filename = f"valhalla_tax_plan_{client_name.replace(' ', '_')}.pdf"
    file_path = os.path.join(temp_dir, filename)

    pdf.output(file_path)

    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename
    )
@app.post("/state_tax_arizona", summary="Estimate Arizona state income tax")
async def state_tax_arizona(data: dict):
    agi = data.get("agi", 0)
    filing_status = data.get("filing_status", "single")
    return calculate_arizona_tax(agi, filing_status)
from fastapi import Body
from state_tax_data import calculate_state_tax

@app.post("/state_tax_estimate", summary="Estimate state tax based on income and state")
async def state_tax_estimate(
    income: float = Body(..., embed=True),
    state: str = Body(..., embed=True)
):
    try:
        result = calculate_state_tax(income, state)
        return {
            "state": state.upper(),
            "income": income,
            "estimated_state_tax": result["state_tax"],
            "effective_rate": result["effective_rate"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/multi_year_roth_projection")
async def multi_year_roth_projection(data: dict):
    agi = data.get("current_agi", 0)
    contributions = data.get("roth_contributions", [])
    filing_status = data.get("filing_status", "single")

    brackets = {
        "single": [
            (0, 11000, "10%"),
            (11001, 44725, "12%"),
            (44726, 95375, "22%"),
            (95376, 182100, "24%"),
            (182101, 231250, "32%"),
            (231251, 578125, "35%"),
            (578126, float("inf"), "37%"),
        ],
        "married_filing_jointly": [
            (0, 22000, "10%"),
            (22001, 89450, "12%"),
            (89451, 190750, "22%"),
            (190751, 364200, "24%"),
            (364201, 462500, "32%"),
            (462501, 693750, "35%"),
            (693751, float("inf"), "37%"),
        ]
    }

    selected = brackets.get(filing_status.lower(), brackets["single"])
    results = []

    for i, amount in enumerate(contributions):
        year = 2025 + i
        new_agi = agi + amount

        for start, end, rate in selected:
            if start <= new_agi <= end:
                results.append({
                    "year": year,
                    "original_agi": agi,
                    "conversion_amount": amount,
                    "new_agi": new_agi,
                    "marginal_rate": rate,
                    "bracket": f"${start:,} – ${end:,}",
                })
                break

    return {"projection": results}
@app.post("/capital_gains_projection")
def capital_gains_projection(data: dict):
    ordinary_income = data.get("ordinary_income", 0)
    capital_gains = data.get("capital_gains", 0)
    filing_status = data.get("filing_status", "single")

    # Example 2025 capital gains brackets
    thresholds = {
        "single": [(0, 44725, "0%"), (44726, 492300, "15%"), (492301, float("inf"), "20%")],
        "married_filing_jointly": [(0, 89450, "0%"), (89451, 553850, "15%"), (553851, float("inf"), "20%")]
    }

    selected = thresholds.get(filing_status.lower(), thresholds["single"])
    taxable = ordinary_income + capital_gains

    for start, end, rate in selected:
        if start <= taxable <= end:
            return {
                "filing_status": filing_status,
                "ordinary_income": ordinary_income,
                "capital_gains": capital_gains,
                "estimated_tax": capital_gains * float(rate.strip('%')) / 100,
                "capital_gains_rate": rate
            }
@app.post("/schedule_c_projection")
async def schedule_c_projection(data: dict):
    revenue = data.get("revenue", 0)
    expenses = data.get("expenses", 0)
    profit = revenue - expenses

    # Simple self-employment tax + income tax example
    se_tax = profit * 0.153  # ~15.3% FICA
    income_tax = profit * 0.22  # assume 22% rate
    total_tax = round(se_tax + income_tax, 2)

    return {
        "profit": profit,
        "self_employment_tax": round(se_tax, 2),
        "estimated_income_tax": round(income_tax, 2),
        "total_tax": total_tax
    }
@app.post("/rental_analysis")
def rental_analysis(data: dict):
    rental_income = data.get("rental_income", 0) or 0
    expenses = data.get("expenses", 0) or 0
    mortgage_interest = data.get("mortgage_interest", 0) or 0
    property_tax = data.get("property_tax", 0) or 0
    insurance = data.get("insurance", 0) or 0
    repairs = data.get("repairs", 0) or 0
    purchase_price = data.get("purchase_price", 0) or 0
    land_value = data.get("land_value", 0) or 0
    filing_status = data.get("filing_status", "single")
    active_participation = data.get("active_participation", True)

    # Depreciation
    depreciable_basis = purchase_price - land_value
    annual_depreciation = depreciable_basis / 27.5 if depreciable_basis > 0 else 0

    total_expenses = expenses + mortgage_interest + property_tax + insurance + repairs + annual_depreciation
    cash_flow = rental_income - (expenses + mortgage_interest + property_tax + insurance + repairs)
    taxable_income = rental_income - total_expenses

    # Passive loss warning logic
    passive_loss_warning = ""
    if taxable_income < 0 and not active_participation:
        passive_loss_warning = "Loss may be limited due to passive activity rules."

    return {
        "rental_income": rental_income,
        "cash_flow": cash_flow,
        "taxable_income": taxable_income,
        "annual_depreciation": annual_depreciation,
        "passive_loss_warning": passive_loss_warning
    }
from fastapi import Body

@app.post("/retirement_contribution_projection")
def retirement_contribution_projection(
    current_agi: float = Body(...),
    contribution_amount: float = Body(...),
    filing_status: str = Body(...)
):
    # Assume contributions are fully deductible
    new_agi = current_agi - contribution_amount

    # Simplified bracket ranges (2025 estimates)
    brackets = {
        "single": [
            (0, 11000, 0.10),
            (11001, 44725, 0.12),
            (44726, 95375, 0.22),
            (95376, 182100, 0.24),
        ],
        "married_filing_jointly": [
            (0, 22000, 0.10),
            (22001, 89450, 0.12),
            (89451, 190750, 0.22),
            (190751, 364200, 0.24),
        ]
    }

    def find_bracket(agi):
        for low, high, rate in brackets.get(filing_status, []):
            if low <= agi <= high:
                return f"${low:,} – ${high:,}", rate
        return "Over highest bracket", 0.32

    original_bracket, original_rate = find_bracket(current_agi)
    new_bracket, new_rate = find_bracket(new_agi)

    tax_savings = contribution_amount * original_rate  # Approximate

    return {
        "original_agi": current_agi,
        "new_agi": new_agi,
        "contribution": contribution_amount,
        "tax_savings": tax_savings,
        "original_bracket": original_bracket,
        "new_bracket": new_bracket,
        "marginal_rate": f"{int(original_rate * 100)}%",
    }
@app.post("/year_end_plan")
async def year_end_plan(input: YearEndPlanInput):
    try:
        agi = (
            input.w2_income
            + input.business_income
            + input.capital_gains
            - input.retirement_contributions
            - input.hsa_contributions
        )
        taxable_income = max(0, agi - input.itemized_deductions)
        estimated_tax = round(taxable_income * 0.22, 2)  # Simplified estimate

        strategies = []
        if input.retirement_contributions < 23000:
            strategies.append("Increase 401(k) contributions to reduce taxable income.")
        if input.hsa_contributions < 8300:
            strategies.append("Maximize HSA contributions for additional deduction.")
        if input.itemized_deductions < 13000:
            strategies.append("Explore bunching deductions before year-end.")
        if input.estimated_payments < estimated_tax:
            strategies.append("Make an estimated tax payment to avoid penalties.")

        return {
            "agi": round(agi, 2),
            "taxable_income": round(taxable_income, 2),
            "estimated_tax": estimated_tax,
            "strategies": strategies,
            "year_end_deadline": "December 31, 2025",
        }
    except Exception as e:
        return {"error": str(e)}
@app.post("/quick_entry_plan")
async def quick_entry_plan(data: dict):
    filing_status = data.get("filing_status", "single")
    w2_income = data.get("w2_income", 0)
    business_income = data.get("business_income", 0)
    capital_gains = data.get("capital_gains", 0)
    dividend_income = data.get("dividend_income", 0)
    retirement_contributions = data.get("retirement_contributions", 0)

    agi = (
        w2_income
        + business_income
        + capital_gains
        + dividend_income
        - retirement_contributions
    )

    # Step 1 – Run threshold modeling
    threshold_result = threshold_modeling({
        "filing_status": filing_status,
        "agi": agi,
        "magi": agi
    })

    # Step 2 – Recommend strategies
    strategy_result = recommend({
        "agi": agi,
        "filing_status": filing_status,
        "business_income": data.get("business_income", 0),
        "retirement_plan_type": "401k"
    })

    # Step 3 – Build PDF payload
    taxable_income = max(0, agi - data.get("itemized_deductions", 0))
    estimated_tax = round(taxable_income * 0.22, 2)
    pdf_payload = {
        "filing_status": filing_status,
        "agi": agi,
        "taxable_income": taxable_income,
        "total_tax": estimated_tax,
        "marginal_rate": "22%",
        "strategies": strategy_result["strategies"],
        "comparison_chart_data": {
            "labels": ["AGI", "Est. Tax"],
            "values": [agi, estimated_tax]
        }
    }

    # Step 4 – Generate PDF
    pdf_bytes = generate_tax_plan_pdf(pdf_payload)

    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf")
# === Smart Strategy PDF Report Handler ===
async def smart_strategy_report(data):
    required_fields = ["agi", "taxable_income", "total_tax", "filing_status"]

    def _has_value(value):
        return value not in (None, "", "N/A")

    # Prefer parsed payloads when present, but allow direct extracted fields from GPT.
    parsed_data = None
    possible_parsed_payloads = [
        data.get("parsed_1040"),
        data.get("parse_1040_result"),
        data.get("parsed_data"),
        data if data.get("status") == "success" else None,
    ]
    for payload in possible_parsed_payloads:
        if isinstance(payload, dict) and payload.get("status") == "success":
            parsed_data = payload
            break

    source_data = parsed_data if parsed_data is not None else data
    received_fields = [field for field in required_fields if _has_value(source_data.get(field))]
    missing_fields = [field for field in required_fields if field not in received_fields]
    if missing_fields:
        response = {
            "status": "insufficient_data",
            "message": "Missing core tax inputs required for planning.",
            "missing_fields": missing_fields,
        }
        if source_data.get("confidence_engine") is not None:
            response["confidence_engine"] = source_data.get("confidence_engine")
        if source_data.get("planning_status") is not None:
            response["planning_status"] = source_data.get("planning_status")
        return response

    agi = source_data.get("agi", 0)
    filing_status = source_data.get("filing_status", "single")
    taxable_income = source_data.get("taxable_income", max(0, (agi or 0) - 13000))
    total_tax = source_data.get("total_tax", source_data.get("estimated_tax", 0))
    marginal_rate = source_data.get("marginal_rate", "22%")

    strategy_result = await recommend({
        "agi": agi,
        "filing_status": filing_status,
        "business_income": source_data.get("business_income", 0),
        "retirement_plan_type": "401k"
    })

    threshold_result = await threshold_modeling({
        "agi": agi,
        "magi": agi,
        "filing_status": filing_status
    })
    estimated_tax = threshold_result.get("estimated_tax", total_tax)
    top_rate = 0.22 if "22" in str(marginal_rate) else 0.24
    strategy_lines = []
    def _fmt_currency(value):
        try:
            return f"${float(value):,.2f}"
        except (TypeError, ValueError):
            return "N/A"

    for strategy in strategy_result.get("strategies", []):
        lowered = strategy.lower()
        if any(keyword in lowered for keyword in ["401(k)", "solo 401", "sep", "ira", "retirement"]):
            sample_contribution = min(5000, max(0, agi * 0.05))
            est_savings = sample_contribution * top_rate
            strategy_lines.append(
                f"- {strategy}\n  Example: Contributing {_fmt_currency(sample_contribution)} to pre-tax retirement can reduce current federal income tax by approximately {_fmt_currency(est_savings)} at a {int(top_rate*100)}% marginal rate. Impact: lower current-year income tax while increasing long-term retirement assets. Note: retirement contributions are not treated here as self-employment tax savings."
            )
        elif "schedule c" in lowered or ("business" in lowered and "deduction" in lowered):
            sample_deduction = 5000
            income_tax_savings = sample_deduction * top_rate
            se_tax_savings = sample_deduction * 0.153
            strategy_lines.append(
                f"- {strategy}\n  Example: A {_fmt_currency(sample_deduction)} Schedule C deduction may reduce income tax by about {_fmt_currency(income_tax_savings)} ({int(top_rate*100)}% marginal rate) plus about {_fmt_currency(se_tax_savings)} in self-employment tax (15.3% planning estimate). Impact: lowers both income-tax and SE-tax exposure when the deduction is business-related."
            )
        elif "HSA" in strategy:
            hsa_add = 3000
            hsa_savings = hsa_add * top_rate
            strategy_lines.append(
                f"- {strategy}\n  Example: An additional {_fmt_currency(hsa_add)} HSA contribution may save roughly {_fmt_currency(hsa_savings)} in federal tax. Impact: immediate deduction plus tax-free medical reimbursement potential."
            )
        elif "estimated tax payment" in lowered:
            projected_shortfall = max(0, (estimated_tax or 0) - source_data.get("estimated_payments", 0))
            strategy_lines.append(
                f"- {strategy}\n  Example: If projected tax is {_fmt_currency(estimated_tax)} and paid-in amounts are {_fmt_currency(source_data.get('estimated_payments', 0))}, a catch-up payment of about {_fmt_currency(projected_shortfall)} can reduce underpayment risk. Impact: avoids penalties and smooths cash flow."
            )
        else:
            strategy_lines.append(
                f"- {strategy}\n  Example: Apply this strategy at the current marginal rate of {marginal_rate} to prioritize dollars that would otherwise be taxed at the highest current bracket. Impact: targeted tax reduction with measurable annual benefit."
            )

    return {
        "report_text": "\n\n".join(strategy_lines) if strategy_lines else "No tax strategies were generated from the current input data.",
        "strategy_count": len(strategy_lines),
        "threshold_flags": threshold_result.get("threshold_flags", []),
        "estimated_tax": estimated_tax,
        "taxable_income": taxable_income,
        "total_tax": total_tax
    }
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

class StrategyROIInput(BaseModel):
    filing_status: str
    w2_income: float
    business_income: float
    capital_gains: float
    dividend_income: float
    retirement_contributions: float
    itemized_deductions: float
    estimated_payments: float
    state: Optional[str] = "AZ"
    show_pdf: Optional[bool] = False
    strategy_flags: Optional[List[str]] = [
        "roth_conversion", "s_corp_election", "aca_optimization"
    ]

from fastapi.responses import FileResponse
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas
import tempfile

@app.post("/generate_strategy_with_roi")
def generate_strategy_with_roi(data: StrategyROIInput):
    agi = (
        data.w2_income +
        data.business_income +
        data.capital_gains +
        data.dividend_income -
        data.retirement_contributions
    )
    standard_deduction = 15000 if data.filing_status == "single" else 30000
    deduction = max(standard_deduction, data.itemized_deductions)
    taxable_income = max(0, agi - deduction)

    strategies = []
    conflicts = []
    priority_recommendation = "No priority recommendation generated yet."
    ranked_recommendations = []
    threshold_flags = []
    if "roth_conversion" in data.strategy_flags and data.business_income > 0:
        roth_tax_cost = 0.22 * data.business_income
        roth_future_savings = roth_tax_cost * 2.5
        strategies.append({
            "name": "Roth Conversion",
            "tax_cost": round(roth_tax_cost, 2),
            "roi": round(roth_future_savings - roth_tax_cost, 2),
            "summary": f"Convert ${data.business_income} to Roth. Pay ${roth_tax_cost:.2f} now, potentially save ${roth_future_savings:.2f} long-term."
        })

    if "s_corp_election" in data.strategy_flags and data.business_income > 30000:
        payroll = 0.6 * data.business_income
        se_tax_savings = 0.153 * (data.business_income - payroll)
        strategies.append({
            "name": "S-Corp Election",
            "tax_cost": 0,
            "roi": round(se_tax_savings, 2),
            "summary": f"Elect S-Corp. Reasonable salary: ${payroll:.0f}. Estimated self-employment tax savings: ${se_tax_savings:.2f}."
        })

    if "aca_optimization" in data.strategy_flags and taxable_income < 75000:
        subsidy_value = 3200
        strategies.append({
            "name": "ACA Subsidy Preservation",
            "tax_cost": 0,
            "roi": subsidy_value,
            "summary": f"Retain estimated ACA subsidy of ${subsidy_value:.2f} by keeping income under threshold."
        })

    # ---- Bracket Intelligence ----
    if data.filing_status == "single":
        if taxable_income < 103350:
            room = 103350 - taxable_income
            strategies.append({
                "name": "Bracket Optimization",
                "tax_cost": 0,
                "roi": 0,
                "summary": f"You have approximately ${room:,.0f} of room remaining in the 22% bracket before entering the 24% bracket. This creates an opportunity for Roth conversions or capital gain harvesting."
            })

    elif data.filing_status in ["married_filing_jointly", "mfj"]:
        if taxable_income < 206700:
            room = 206700 - taxable_income
            strategies.append({
                "name": "Bracket Optimization",
                "tax_cost": 0,
                "roi": 0,
                "summary": f"You have approximately ${room:,.0f} of room remaining in the 22% bracket before entering the 24% bracket. This creates an opportunity for income acceleration strategies."
            })
    # ---- Threshold / Phaseout Detection ----
    if data.filing_status == "single":
        if agi > 161000:
            threshold_flags.append(
                "Roth IRA contribution phaseout may apply because AGI is above the Single filer threshold."
            )

        if agi > 200000:
            threshold_flags.append(
                "NIIT risk: Net Investment Income Tax may apply above $200,000 AGI for Single filers."
            )

        if agi > 60000 and "aca_optimization" in data.strategy_flags:
            threshold_flags.append(
                "ACA subsidy risk: income may reduce premium tax credits. Avoid Roth conversions or capital gains without modeling subsidy impact first."
            )

    elif data.filing_status in ["married_filing_jointly", "mfj"]:
        if agi > 240000:
            threshold_flags.append(
                "Roth IRA contribution phaseout may apply for MFJ."
            )

        if agi > 250000:
            threshold_flags.append(
                "NIIT risk: Net Investment Income Tax may apply above $250,000 AGI for MFJ."
            )

        if agi > 80000 and "aca_optimization" in data.strategy_flags:
            threshold_flags.append(
                "ACA subsidy risk: income may reduce or eliminate premium tax credits."
            )         
    # ---- Strategy Conflict Detection ----
    if "roth_conversion" in data.strategy_flags and "aca_optimization" in data.strategy_flags:
        conflicts.append(
            "Conflict detected: Roth conversions increase income and may reduce ACA subsidies."
        )

    if "s_corp_election" in data.strategy_flags and data.retirement_contributions > 0:
        conflicts.append(
            "Conflict detected: S-Corp salary structure can reduce retirement contribution efficiency."
        )

    if data.capital_gains > 0 and "aca_optimization" in data.strategy_flags:
        conflicts.append(
            "Conflict detected: Capital gains increase income and may reduce ACA subsidy eligibility."
        )
   # ---- Priority Recommendation Engine ----
    if conflicts:
        priority_recommendation = (
            "Review strategy conflicts before implementation. Do not automatically pursue every strategy at once; prioritize the strategy that best matches the client's main goal."
        )

    elif data.business_income > 30000 and "s_corp_election" in data.strategy_flags:
        priority_recommendation = (
            "Primary recommendation: review S-Corp election first because business income may create self-employment tax savings opportunities."
        )

    elif "aca_optimization" in data.strategy_flags and taxable_income < 75000:
        priority_recommendation = (
            "Primary recommendation: preserve ACA subsidy eligibility first because increasing income may reduce or eliminate valuable premium tax credits."
        )

    elif data.filing_status == "single" and taxable_income < 103350:
        priority_recommendation = (
            "Primary recommendation: use remaining 22% bracket room strategically through Roth conversion planning, capital gain harvesting, or controlled income timing."
        )

    else:
        priority_recommendation = (
            "Primary recommendation: complete a withholding review and compare current-year income, deductions, and tax payments before implementing advanced strategies."
        )

    # ---- Dollar Impact Ranking ----
    ranked_recommendations = sorted(
        strategies,
        key=lambda s: float(s.get("roi", 0) or 0),
        reverse=True
    )
    # ---- Total Estimated Planning Value ----
    total_estimated_roi = sum(
        float(s.get("roi", 0) or 0)
        for s in strategies
    )

    # ---- Scenario Comparison ----
    # ---- Progressive Federal Tax Estimate ----
    if data.filing_status == "single":
        brackets = [
            (0, 11925, 0.10),
            (11925, 48475, 0.12),
            (48475, 103350, 0.22),
            (103350, 197300, 0.24),
            (197300, 250525, 0.32),
            (250525, 626350, 0.35),
            (626350, float("inf"), 0.37),
        ]
    else:
        brackets = [
            (0, 23850, 0.10),
            (23850, 96950, 0.12),
            (96950, 206700, 0.22),
            (206700, 394600, 0.24),
            (394600, 501050, 0.32),
            (501050, 751600, 0.35),
            (751600, float("inf"), 0.37),
        ]

    baseline_tax = 0

    for lower, upper, rate in brackets:
        if taxable_income > lower:
            taxed_amount = min(taxable_income, upper) - lower
            baseline_tax += taxed_amount * rate
    optimized_tax = baseline_tax - total_estimated_roi
    tax_savings = baseline_tax - optimized_tax

    # ---- Client Action Steps ----
    action_steps = []

    if ranked_recommendations:
        action_steps.append(
            f"Start with: {ranked_recommendations[0].get('name', 'Top strategy')}."
        )

    if conflicts:
        action_steps.append(
            "Review the listed conflicts before implementing any strategy."
        )

    if data.business_income > 0:
        action_steps.append(
            "Gather Schedule C/K-1 business income details, deductible expenses, and entity records."
        )

    if data.retirement_contributions == 0:
        action_steps.append(
            "Review available retirement plan options and contribution limits before year-end."
        )

    action_steps.append(
        "Run a final tax projection before implementing the recommended strategy."
    )
    # ---- Client Report Text ----
    strategy_lines = ""
    for index, strategy in enumerate(ranked_recommendations, start=1):
        strategy_lines += (
            f"{index}. {strategy.get('name', 'Strategy')}\n"
            f"   Estimated Value: ${float(strategy.get('roi', 0) or 0):,.2f}\n"
            f"   Summary: {strategy.get('summary', '')}\n\n"
        )

    conflict_lines = "\n".join(f"- {c}" for c in conflicts) if conflicts else "No major strategy conflicts identified."

    threshold_lines = "\n".join(f"- {t}" for t in threshold_flags) if threshold_flags else "No major threshold or phaseout risks identified."

    action_lines = "\n".join(f"- {a}" for a in action_steps)

    client_report_text = (
        "STRATEGIC TAX PLANNING REPORT\n\n"
        f"Primary Recommendation:\n{priority_recommendation}\n\n"
        "CURRENT TAX POSITION\n"
        f"Adjusted Gross Income: ${agi:,.2f}\n"
        f"Taxable Income: ${taxable_income:,.2f}\n"
        f"Estimated Federal Tax Before Planning: ${baseline_tax:,.2f}\n"
        f"Estimated Federal Tax After Planning: ${optimized_tax:,.2f}\n"
        f"Estimated Tax Savings: ${tax_savings:,.2f}\n\n"
        "RECOMMENDED STRATEGIES\n"
        f"{strategy_lines}\n"
        "CONFLICT AND RISK ANALYSIS\n"
        f"{conflict_lines}\n\n"
        "THRESHOLD AND PHASEOUT REVIEW\n"
        f"{threshold_lines}\n\n"
        "CLIENT ACTION STEPS\n"
        f"{action_lines}\n\n"
        "FINAL RECOMMENDATION\n"
        "The recommended next step is to review the highest-impact strategy first, "
        "confirm the supporting income and deduction details, and complete a final tax projection before implementation.\n\n"
        "Prepared by Valhalla Tax Services"
    )
    return {
        "agi": round(agi, 2),
        "taxable_income": round(taxable_income, 2),
        "strategies": strategies,
        "conflicts": conflicts,
        "threshold_flags": threshold_flags,
        "priority_recommendation": priority_recommendation,
        "ranked_recommendations": ranked_recommendations,
        "total_estimated_roi": round(total_estimated_roi, 2),
        "action_steps": action_steps,
        "baseline_tax": round(baseline_tax, 2),
        "optimized_tax": round(optimized_tax, 2),
        "tax_savings": round(tax_savings, 2),
        "client_report_text": client_report_text
    }
@app.get("/action_ping")
async def action_ping():
    print("====== ACTION PING HIT ======", flush=True)
    return {"status": "ok", "message": "GPT action reached Render"}
