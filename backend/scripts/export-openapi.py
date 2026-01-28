"""
Export OpenAPI spec from FastAPI app to JSON file.

Usage: python scripts/export-openapi.py [output_path]
"""
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

def export_openapi(output_path: str = "openapi.json"):
    spec = app.openapi()
    with open(output_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"OpenAPI spec exported to {output_path}")

if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "openapi.json"
    export_openapi(output)
