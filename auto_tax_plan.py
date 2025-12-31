from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from parse_1040 import parse1040
from parse_1120s import parse1120s
from generate_strategy_with_roi import generateStrategyWithROI

router = APIRouter()

@router.post("/auto_tax_plan")
async def auto_tax_plan(file: UploadFile = File(...)):
    print(f"📥 Received file: {file.filename}")  # <--- LOGGING LINE HERE

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

