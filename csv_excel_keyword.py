
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import io
import os
from pathlib import Path
import PyPDF2

router = APIRouter()

@router.post("/parse_csv_excel")
async def parse_csv_excel(file: UploadFile = File(...)):
    contents = await file.read()
    file_type = file.filename.split(".")[-1].lower()

    if file_type == "csv":
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    elif file_type in ["xls", "xlsx"]:
        df = pd.read_excel(io.BytesIO(contents))
    else:
        return {"error": "Unsupported file type"}

    return {
        "summary": f"Parsed {len(df)} rows and {len(df.columns)} columns",
        "columns": df.columns.tolist(),
        "rows": df.head(10).values.tolist()
    }

@router.post("/keyword_detection")
async def keyword_detection(file: UploadFile = File(...), keywords: list[str] = []):
    content = ""
    if file.filename.lower().endswith(".pdf"):
        reader = PyPDF2.PdfReader(file.file)
        content = "".join(page.extract_text() or "" for page in reader.pages)
    else:
        contents = await file.read()
        try:
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        except:
            df = pd.read_excel(io.BytesIO(contents))
        content = df.to_string()

    matches = [kw for kw in keywords if kw.lower() in content.lower()]
    context = []
    for kw in matches:
        index = content.lower().find(kw.lower())
        snippet = content[max(index-30,0):index+70]
        context.append(snippet)

    return {
        "matched_terms": matches,
        "context": context
    }

@router.post("/preview_valhalla_premium_strategy")
def preview_valhalla_premium_strategy(payload: dict):
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

@router.post("/generate_valhalla_premium_docx")
def generate_valhalla_premium_docx(payload: dict):
    try:
        from valhalla_premium_docx_report import generate_valhalla_docx_report

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
