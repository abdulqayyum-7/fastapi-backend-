#!/bin/bash
# Activate virtual environment if you have one
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI app
uvicorn app.main:app --host 0.0.0.0 --port $PORT
