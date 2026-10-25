import sys
from pathlib import Path

# Redirect entry point to frontend/app.py
frontend_app = Path(__file__).resolve().parent / "frontend" / "app.py"

with open(frontend_app, "r", encoding="utf-8") as f:
    code = f.read()

exec(code)
