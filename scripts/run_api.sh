#!/usr/bin/env bash
source .venv/bin/activate
uvicorn app.api:app --host 0.0.0.0 --port 8000