import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import torch

# ---- PERFORMANCE TUNING (important for Render) ----
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
# -----------------------------------------------

from model_utils import predict_audio_from_base64

load_dotenv()
app = FastAPI()

API_KEY = os.getenv("API_KEY")

# -------- REQUEST MODEL (Hackathon format) --------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str
# -----------------------------------------------

@app.get("/")
def server():
    return {"message": "server is running"}

# ============ ONLY API YOU NEED ============
@app.post("/voice-detection")
def voice_detection(
    payload: VoiceRequest,
    x_api_key: str = Header(None, alias="x-api-key")
):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if payload.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 supported")

    classification, confidence = predict_audio_from_base64(
        payload.audioBase64
    )

    explanation = (
        "Unnatural pitch consistency and robotic speech patterns detected"
        if classification == "AI_GENERATED"
        else "Natural speech variations and human-like intonation detected"
    )

    return {
        "status": "success",
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
