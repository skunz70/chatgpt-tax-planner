
from fastapi import APIRouter, UploadFile, File
import pandas as pd
import io
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
