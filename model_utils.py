import torch
import torch.nn as nn
import numpy as np
from audio_utils import decode_base64_to_audio, extract_melspectrogram
import soundfile as sf
# ===========================================================
# 1️⃣ YOUR EXACT TRAINING MODEL (no placeholders anymore)
# ===========================================================

class HybridCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        # Dynamic FC (same as your training code)
        self.fc = None  

        self.ai_head = nn.Linear(64, 1)
        self.lang_head = nn.Linear(64, 5)   # NOTE: you trained with 5 languages

    def forward(self, x):
        x = self.features(x)

        # flatten dynamically
        x = torch.flatten(x, start_dim=1)

        if self.fc is None:
            self.fc = nn.Sequential(
                nn.Linear(x.shape[1], 64),
                nn.ReLU(),
                nn.Dropout(0.5)
            ).to(x.device)

        x = self.fc(x)

        ai_out = torch.sigmoid(self.ai_head(x))
        lang_out = self.lang_head(x)

        return ai_out, lang_out


# ===========================================================
# 2️⃣ LOAD YOUR WEIGHTS CORRECTLY
# ===========================================================

model = HybridCNN()

dummy = torch.randn(1, 1, 128, 157)   # ← change if needed
with torch.no_grad():
    model(dummy)

state_dict = torch.load(
    "pytorch_voice_model.pt",
    map_location="cpu",
    weights_only=True
)   

model.load_state_dict(state_dict)
model.eval()

# Your language mapping (must match training)
language_map = {
    0: "hindi",
    1: "tamil",
    2: "telugu",
    3: "malayalam",
    4: "english"      # IMPORTANT: your model has 5 classes
}

# ===========================================================
# 3️⃣ PREDICTION FUNCTION (unchanged logic)
# ===========================================================

def predict_audio_from_base64(audio_b64):

    audio, sr = decode_base64_to_audio(audio_b64)

    spec = extract_melspectrogram(audio, sr)

    # (B, C, H, W)
    spec = np.expand_dims(spec, axis=0)
    spec = np.expand_dims(spec, axis=0)
    spec_tensor = torch.tensor(spec, dtype=torch.float32)

    with torch.no_grad():
        ai_prob, lang_logits = model(spec_tensor)

    ai_prob = ai_prob.numpy()
    lang_probs = torch.softmax(lang_logits, dim=1).numpy()

    ai_label = "AI_GENERATED" if ai_prob[0][0] > 0.5 else "HUMAN"

    lang_id = np.argmax(lang_probs[0])
    language = language_map[lang_id]

    return ai_label, float(ai_prob[0][0]), language


def predict_audio_from_file(file_obj):

    # Read audio from uploaded file
    audio, sr = sf.read(file_obj)

    # Convert to mel spectrogram
    spec = extract_melspectrogram(audio, sr)

    # Make shape: (1, 1, H, W)
    spec = np.expand_dims(spec, axis=0)
    spec = np.expand_dims(spec, axis=0)
    spec_tensor = torch.tensor(spec, dtype=torch.float32)

    with torch.no_grad():
        ai_prob, lang_logits = model(spec_tensor)

    ai_prob = ai_prob.numpy()
    lang_probs = torch.softmax(lang_logits, dim=1).numpy()

    ai_label = "AI_GENERATED" if ai_prob[0][0] > 0.5 else "HUMAN"

    lang_id = np.argmax(lang_probs[0])
    language = language_map[lang_id]

    return ai_label, float(ai_prob[0][0]), language