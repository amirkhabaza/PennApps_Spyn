# server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import os
import requests
import io

app = FastAPI()

# ElevenLabs API configuration
ELEVENLABS_API_KEY = "Ap2_1039e172-f84f-4874-b905-a668796d765f"
ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam voice (default)
ELEVENLABS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

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

@app.post("/speak")
async def speak_text(request: dict):
    """
    Convert text to speech using ElevenLabs API.
    """
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Prepare request to ElevenLabs API
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        # Make request to ElevenLabs
        response = requests.post(ELEVENLABS_URL, json=data, headers=headers)
        
        if response.status_code == 200:
            # Return audio as streaming response
            audio_content = response.content
            return StreamingResponse(
                io.BytesIO(audio_content),
                media_type="audio/mpeg",
                headers={"Content-Disposition": "inline; filename=speech.mp3"}
            )
        else:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"ElevenLabs API error: {response.text}"
            )
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/")
def root():
    return {"status": "API is running", "endpoint": "/metrics"}
