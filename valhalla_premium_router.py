from fastapi import APIRouter, HTTPException, Request
from pathlib import Path
import os

router = APIRouter()


def normalize_payload(payload: dict) -> dict:
    data = payload or {}

    return {
        "client_name": data.get("client_name", "Client"),
        "tax_year": data.get("tax_year", 2025),
        "filing_status": data.get("filing_status", "MFJ"),
        "agi": data.get("agi", 0),
        "taxable_income": data.get("taxable_income", 0),
        "total_tax": data.get("total_tax", 0),
        "federal_withholding": data.get("federal_withholding", 0),
        "schedule_c_gross_revenue": data.get("schedule_c_gross_revenue", 0),
        "schedule_c_net_profit": data.get("schedule_c_net_profit", 0),
        "contract_labor": data.get("contract_labor", 0),
        "capital_gains": data.get("capital_gains", 0),
        "age": data.get("age", 0),
        "state": data.get("state", "AZ"),
        "has_retirement_accounts": data.get("has_retirement_accounts", False),
        "logo_path": data.get("logo_path", "valhalla_logo.jpg"),
    }


@router.post("/generate_valhalla_premium_docx")
def generate_valhalla_premium_docx(payload: dict, request: Request):

    try:
        from valhalla_premium_docx_report import generate_valhalla_docx_report

        normalized = normalize_payload(payload)

        client_name = str(
            normalized.get("client_name", "client")
        ).replace(" ", "_")

        tax_year = str(normalized.get("tax_year", "2025"))

        output_dir = Path("generated_reports")
        output_dir.mkdir(exist_ok=True)

        output_path = (
            output_dir /
            f"valhalla_premium_{client_name}_{tax_year}.docx"
        )

        generated_path = generate_valhalla_docx_report(
            normalized,
            str(output_path)
        )

        if not os.path.exists(generated_path):
            raise HTTPException(
                status_code=500,
                detail="DOCX report was not created."
            )

        base_url = str(request.base_url).rstrip("/")
        filename = os.path.basename(generated_path)
        download_url = f"{base_url}/generated_reports/{filename}"

        return {
            "status": "success",
            "message": "Valhalla Premium DOCX report generated successfully.",
            "filename": filename,
            "download_url": download_url
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )