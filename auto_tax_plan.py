from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Body
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import os
import re

from parse_1040 import parse1040
from parse_1120s import parse1120s
from generate_strategy_with_roi import generateStrategyWithROI

router = APIRouter()

DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
REPORT_DIR = Path("generated_reports")


@router.post("/auto_tax_plan")
async def auto_tax_plan(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")

    # Save uploaded temporary PDF
    temp_name = f"/tmp/{file.filename}"
    with open(temp_name, "wb") as f:
        f.write(await file.read())

    # Guess form type by filename or first page text
    if "1040" in file.filename:
        parsed = parse1040(temp_name)
    elif "1120" in file.filename:
        parsed = parse1120s(temp_name)
    else:
        raise HTTPException(status_code=400, detail="Unsupported tax form")

    # Generate strategy
    strat = generateStrategyWithROI(parsed)

    return JSONResponse(content=strat)


def _normalize_valhalla_payload(payload: dict) -> dict:
    data = payload or {}
    return {
        "client_name": data.get("client_name", "Test Client"),
        "tax_year": data.get("tax_year", 2025),
        "filing_status": data.get("filing_status", "MFJ"),
        "agi": data.get("agi", 185000),
        "taxable_income": data.get("taxable_income", 142000),
        "total_tax": data.get("total_tax", 24000),
        "refund": data.get("refund", 1200),
        "balance_due": data.get("balance_due", 0),
        "federal_withholding": data.get("federal_withholding", 22800),
        "dependents": data.get("dependents", 2),
        "state": data.get("state", "AZ"),
        "schedule_c_gross_revenue": data.get("schedule_c_gross_revenue", 280000),
        "schedule_c_net_profit": data.get("schedule_c_net_profit", 95000),
        "contract_labor": data.get("contract_labor", 107920),
        "w2_income": data.get("w2_income", 90000),
        "capital_gains": data.get("capital_gains", 18000),
        "dividend_income": data.get("dividend_income", 3500),
        "interest_income": data.get("interest_income", 1200),
        "has_retirement_accounts": data.get("has_retirement_accounts", False),
        "age": data.get("age", 64),
        "aca_marketplace": data.get("aca_marketplace", False),
        "logo_path": data.get("logo_path", "valhalla_logo.jpg"),
    }


def _json_docx_response(
    status: str,
    message: str,
    filename: str = "",
    content_type: str = DOCX_CONTENT_TYPE,
    download_url: str = "",
    download_path: str = "",
) -> JSONResponse:
    return JSONResponse(
        status_code=200,
        media_type="application/json",
        content={
            "status": str(status),
            "message": str(message),
            "filename": str(filename),
            "content_type": str(content_type),
            "download_url": str(download_url),
            "download_path": str(download_path),
        },
    )


@router.post(
    "/generate_valhalla_premium_docx",
    operation_id="generateValhallaPremiumDocx",
    summary="Generate a premium Valhalla DOCX tax strategy report",
)
def generate_valhalla_premium_docx(
    request: Request,
    payload: dict | None = Body(default=None),
):
    try:
        from valhalla_premium_docx_report import generate_valhalla_docx_report

        normalized = _normalize_valhalla_payload(payload)
        client_name = str(normalized.get("client_name", "client"))
        client_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", client_name).strip("_") or "client"
        tax_year = str(normalized.get("tax_year", "2025"))
        tax_year = re.sub(r"[^A-Za-z0-9_.-]+", "_", tax_year).strip("_") or "2025"

        REPORT_DIR.mkdir(exist_ok=True)
        output_path = REPORT_DIR / f"valhalla_premium_{client_name}_{tax_year}.docx"
        generated_path = generate_valhalla_docx_report(normalized, str(output_path))

        if not os.path.exists(str(generated_path)):
            return _json_docx_response("error", "DOCX report was not created.")

        filename = os.path.basename(str(generated_path))
        download_path = f"/generated_reports/{filename}"
        download_url = f"{str(request.base_url).rstrip('/')}{download_path}"

        return _json_docx_response(
            status="success",
            message="Valhalla Premium DOCX report generated successfully.",
            filename=filename,
            content_type=DOCX_CONTENT_TYPE,
            download_url=download_url,
            download_path=download_path,
        )
    except Exception as e:
        return _json_docx_response(
            status="error",
            message=f"Valhalla Premium DOCX report generation failed: {e}",
            content_type="application/json",
        )


@router.get("/generated_reports/{filename}", include_in_schema=False)
def download_generated_report(filename: str):
    safe_filename = os.path.basename(filename)
    report_path = REPORT_DIR / safe_filename
    if not report_path.exists() or report_path.suffix.lower() != ".docx":
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(
        path=str(report_path),
        filename=safe_filename,
        media_type=DOCX_CONTENT_TYPE,
    )
