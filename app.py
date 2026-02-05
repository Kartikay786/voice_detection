from fastapi import FastAPI, Header, HTTPException
from model_utils import predict_audio_from_base64
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

API_KEY =  os.getenv("API_KEY")

@app.post("/predict")
def predict(payload: dict, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if "audio_base64" not in payload:
        raise HTTPException(status_code=400, detail="Missing audio_base64")

    audio_b64 = payload["audio_base64"]

    classification, confidence, language = predict_audio_from_base64(audio_b64)

    explanation = (
        "Unnatural pitch consistency and robotic speech patterns detected"
        if classification == "AI_GENERATED"
        else "Natural speech variations and human-like intonation detected"
    )

    return {
        "status": "success",
        "language": language.capitalize(),
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
