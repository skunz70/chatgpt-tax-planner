from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
import os
from uuid import uuid4

router = APIRouter()

@router.post("/auto_tax_plan")
async def auto_tax_plan(file: UploadFile = File(...)):
    # Save uploaded file
    temp_filename = f"/tmp/{uuid4()}_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    # Simulate OCR & tax analysis
    # TODO: Replace this with actual parsing logic for 1040, 1120-S, etc.
    tax_summary_text = f"""
    Tax Plan Summary (Auto Generated):

    - Detected Form: {file.filename}
    - AGI: $152,000
    - Taxable Income: $122,000
    - Suggested Strategies:
      * Roth Conversion: Convert $40,000 → Save $22,000 over time.
      * S-Corp Election: Save ~$2,448 in SE tax.

    This is a sample output.
    """

    # Save the result to a PDF (as plain text for now)
    result_path = f"/tmp/{uuid4()}_tax_plan.txt"
    with open(result_path, "w") as f:
        f.write(tax_summary_text)

    return FileResponse(result_path, filename="automated_tax_plan.txt")
