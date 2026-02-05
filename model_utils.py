import torch
import numpy as np
from audio_utils import decode_base64_to_audio, extract_melspectrogram

# ---- Load PyTorch model ONCE ----
model = torch.load("ai_voice_detector.pt", map_location=torch.device("cpu"))
model.eval()

language_map = {0:"hindi", 1:"tamil", 2:"telugu", 3:"malayalam"}

def predict_audio_from_base64(audio_b64):

    audio, sr = decode_base64_to_audio(audio_b64)

    spec = extract_melspectrogram(audio, sr)

    # convert to tensor: (1, C, H, W)
    spec = np.expand_dims(spec, axis=0)        # add batch dim
    spec = np.expand_dims(spec, axis=0)        # add channel dim
    spec_tensor = torch.tensor(spec, dtype=torch.float32)

    with torch.no_grad():
        ai_prob, lang_probs = model(spec_tensor)

    ai_prob = ai_prob.numpy()
    lang_probs = lang_probs.numpy()

    ai_label = "AI_GENERATED" if ai_prob[0][0] > 0.5 else "HUMAN"

    lang_id = np.argmax(lang_probs[0])
    language = language_map[lang_id]

    return ai_label, float(ai_prob[0][0]), language
