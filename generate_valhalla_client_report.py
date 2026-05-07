import json
import sys
from pathlib import Path
from valhalla_premium_docx_report import generate_valhalla_docx_report


DEFAULT_OUTPUT = "valhalla_client_report.docx"


REQUIRED_FIELDS = [
    "client_name",
    "tax_year",
    "filing_status",
    "agi",
    "taxable_income",
    "total_tax",
]


def load_payload(path: str) -> dict:
    payload_path = Path(path)
    if not payload_path.exists():
        raise FileNotFoundError(f"Payload file not found: {path}")

    with payload_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_payload(data: dict) -> list:
    missing = []
    for field in REQUIRED_FIELDS:
        if field not in data or data.get(field) in (None, ""):
            missing.append(field)
    return missing


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_valhalla_client_report.py client_payload.json [output.docx]")
        sys.exit(1)

    payload_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_OUTPUT

    data = load_payload(payload_file)
    missing = validate_payload(data)
    if missing:
        print("Missing required fields:")
        for field in missing:
            print(f"- {field}")
        sys.exit(1)

    if "logo_path" not in data:
        data["logo_path"] = "valhalla_logo.jpg"

    generated = generate_valhalla_docx_report(data, output_file)
    print(generated)


if __name__ == "__main__":
    main()
