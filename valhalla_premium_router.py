<<<<<<< HEAD
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from valhalla_premium_docx_report import generate_valhalla_docx_report
=======
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import os
>>>>>>> 608fc4a (Add premium Valhalla DOCX router)

router = APIRouter()


<<<<<<< HEAD
@router.post("/generate_valhalla_premium_docx")
def generate_valhalla_premium_docx(payload: dict):
    """
    Generate a premium Valhalla DOCX tax strategy report from structured client data.

    This endpoint is intentionally separate from /gpt-tax-router and legacy PDF routes.
    It uses the premium DOCX engine, dynamic strategy engine, and ROI scoring engine.
    """
    try:
        client_name = str(payload.get("client_name", "client")).replace(" ", "_")
        tax_year = str(payload.get("tax_year", "2025"))
        output_dir = Path("generated_reports")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"valhalla_premium_{client_name}_{tax_year}.docx"

        if "logo_path" not in payload:
            payload["logo_path"] = "valhalla_logo.jpg"

        generated_path = generate_valhalla_docx_report(payload, str(output_path))

        if not os.path.exists(generated_path):
            raise HTTPException(status_code=500, detail="Report generation failed: output file was not created.")
=======
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
def generate_valhalla_premium_docx(payload: dict):

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
>>>>>>> 608fc4a (Add premium Valhalla DOCX router)

        return FileResponse(
            path=generated_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=os.path.basename(generated_path),
        )
<<<<<<< HEAD
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Premium DOCX report generation failed: {str(e)}")


@router.post("/preview_valhalla_premium_strategy")
def preview_valhalla_premium_strategy(payload: dict):
    """
    Lightweight preview endpoint for confirming the premium report payload before generating DOCX.
    """
    try:
        from valhalla_strategy_engine import generate_dynamic_tax_strategy
        from valhalla_roi_engine import generate_roi_analysis

        return {
            "status": "success",
            "client_name": payload.get("client_name", "Client"),
            "tax_year": payload.get("tax_year", "2025"),
            "dynamic_strategy": generate_dynamic_tax_strategy(payload),
            "roi_analysis": generate_roi_analysis(payload),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Premium strategy preview failed: {str(e)}")
=======

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
>>>>>>> 608fc4a (Add premium Valhalla DOCX router)
