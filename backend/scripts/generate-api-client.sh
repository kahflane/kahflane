#!/bin/bash
# Generate C# API client from OpenAPI spec using NSwag.
# Run from backend/ directory.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BACKEND_DIR"

echo "Exporting OpenAPI spec..."
./venv/bin/python scripts/export-openapi.py openapi.json

echo "Generating C# client with NSwag..."
nswag run nswag.json

echo "Done. Client generated at web/Frontend/Generated/KahflaneApiClient.cs"
