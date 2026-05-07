import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from valhalla_premium_docx_report import generate_valhalla_docx_report

router = APIRouter()


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

        return FileResponse(
            path=generated_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=os.path.basename(generated_path),
        )
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
