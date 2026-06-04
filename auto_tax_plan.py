from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Body, Response
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


@router.get("/openapi_final_docx.yaml", include_in_schema=False)
def serve_final_docx_openapi():
    spec_path = Path(__file__).parent / "openapi_final_docx.yaml"
    if not spec_path.is_file():
        return Response("Final DOCX OpenAPI spec not found", media_type="text/plain", status_code=404)
    return Response(spec_path.read_text(encoding="utf-8"), media_type="application/x-yaml")


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


def _first_present(data: dict, *keys, default=None):
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return default


def _normalize_valhalla_payload(payload: dict) -> dict:
    data = payload or {}
    normalized = {
        "client_name": data.get("client_name", "Test Client"),
        "tax_year": data.get("tax_year", 2025),
        "filing_status": data.get("filing_status", "MFJ"),
        "agi": data.get("agi", 185000),
        "taxable_income": data.get("taxable_income", 142000),
        "total_tax": data.get("total_tax", 24000),
        "refund": _first_present(data, "refund", "refund_amount", default=1200),
        "balance_due": _first_present(data, "balance_due", "amount_owed", default=0),
        "federal_withholding": _first_present(data, "federal_withholding", "withholding", "total_withholding", default=22800),
        "dependents": data.get("dependents", 2),
        "state": data.get("state", "AZ"),
        "schedule_c_gross_revenue": _first_present(data, "schedule_c_gross_revenue", "business_gross_revenue", default=280000),
        "schedule_c_net_profit": _first_present(data, "schedule_c_net_profit", "business_income", "business_net_profit", default=95000),
        "contract_labor": data.get("contract_labor", 107920),
        "w2_income": _first_present(data, "w2_income", "wages", default=90000),
        "capital_gains": _first_present(data, "capital_gains", "capital_gain", default=18000),
        "dividend_income": _first_present(data, "dividend_income", "dividends", default=3500),
        "interest_income": _first_present(data, "interest_income", "interest", default=1200),
        "has_retirement_accounts": data.get("has_retirement_accounts", False),
        "age": data.get("age", 64),
        "aca_marketplace": data.get("aca_marketplace", False),
        "logo_path": data.get("logo_path", "valhalla_logo.jpg"),
    }

    # Preserve optional plan narrative fields when the GPT has already produced
    # a reviewed strategy in chat. Missing values never block DOCX generation.
    passthrough_keys = (
        "advisor_summary",
        "top_planning_focus",
        "planning_summary",
        "final_recommendation",
        "tax_efficiency_score",
        "priority_actions",
        "strategies",
        "dynamic_sections",
        "roi_strategies",
    )
    for key in passthrough_keys:
        if key in data and data[key] not in (None, ""):
            normalized[key] = data[key]

    return normalized


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
    operation_id="generateFinalReport",
    summary="Generate the final downloadable client DOCX report",
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