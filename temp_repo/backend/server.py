# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI()

# Allow CORS for frontend on localhost:3000 (React/Next.js) or any other domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev, allow all; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/metrics")
def get_metrics():
    """
    Return latest metrics.json content for frontend.
    """
    path = "./metrics.json"
    if not os.path.exists(path):
        return {"error": "metrics.json not found"}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"status": "API is running", "endpoint": "/metrics"}
