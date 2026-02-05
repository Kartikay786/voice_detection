from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from pydantic import BaseModel
from model_utils import predict_audio_from_file, predict_audio_from_base64
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

API_KEY = os.getenv("API_KEY")

# ----------- REQUEST MODEL FOR BASE64 -----------
class Base64Request(BaseModel):
    audio_base64: str
# -----------------------------------------------
# -------- EXACT FORMAT REQUIRED BY HACKATHON --------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


@app.get('/')
def server():
    return {"message":"server is running"}
# api endpoint for hackathon
@app.post("/voice-detection")
def voice_detection(
    payload: VoiceRequest,
    x_api_key: str = Header(None, alias="x-api-key")   # ← IMPORTANT FIX
):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # You can optionally check format
    if payload.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 supported")

    classification, confidence, detected_language = predict_audio_from_base64(
        payload.audioBase64
    )

    explanation = (
        "Unnatural pitch consistency and robotic speech patterns detected"
        if classification == "AI_GENERATED"
        else "Natural speech variations and human-like intonation detected"
    )

    classification = "AI_GENERATED" if classification == "AI_GENERATED" else "HUMAN_GENERATED"

    return {
        "status": "success",
        # "language": detected_language.capitalize(),
        "classification": classification ,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
# ==============================
#  API 1 — BASE64 METHOD
# ==============================
@app.post("/predict-base64")
def predict_base64(
    payload: Base64Request,
    x_api_key: str = Header(None)
):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    classification, confidence, language = predict_audio_from_base64(
        payload.audio_base64
    )

    explanation = (
        "Unnatural pitch consistency and robotic speech patterns detected"
        if classification == "AI_GENERATED"
        else "Natural speech variations and human-like intonation detected"
    )

    return {
        "status": "success",
        # "language": language.capitalize(),
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }


# ==============================
#  API 2 — FILE UPLOAD METHOD
# ==============================
@app.post("/predict-file")
def predict_file(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    classification, confidence, language = predict_audio_from_file(file.file)

    explanation = (
        "Unnatural pitch consistency and robotic speech patterns detected"
        if classification == "AI_GENERATED"
        else "Natural speech variations and human-like intonation detected"
    )

    return {
        "status": "success",
        # "language": language.capitalize(),
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
