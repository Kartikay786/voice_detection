import torch
import numpy as np
import soundfile as sf
from audio_utils import decode_base64_to_audio, extract_melspectrogram
import torch.nn as nn

# ==========================
# REDEFINE YOUR MODEL EXACTLY
# ==========================

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

        self.fc = None
        self.ai_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)

        if self.fc is None:
            self.fc = nn.Sequential(
                nn.Linear(x.shape[1], 64),
                nn.ReLU(),
                nn.Dropout(0.5)
            ).to(x.device)

        x = self.fc(x)
        ai_out = torch.sigmoid(self.ai_head(x))
        return ai_out

# ==========================
# LOAD YOUR ORIGINAL WEIGHTS
# ==========================

model = HybridCNN()

# create dummy pass so fc layer initializes
dummy = torch.randn(1, 1, 128, 157)
with torch.no_grad():
    model(dummy)

state_dict = torch.load(
    "pytorch_voice_model.pt",
    map_location="cpu",
    weights_only=True
)

# IMPORTANT FIX 👇
model.load_state_dict(state_dict, strict=False)

model = model.to("cpu")
model.eval()


# ==========================
# PREDICTION FUNCTION
# ==========================

def predict_audio_from_base64(audio_b64):

    audio, sr = decode_base64_to_audio(audio_b64)
    spec = extract_melspectrogram(audio, sr)

    spec = np.expand_dims(spec, axis=0)
    spec = np.expand_dims(spec, axis=0)
    spec_tensor = torch.tensor(spec, dtype=torch.float32)

    with torch.no_grad():
        ai_prob = model(spec_tensor)

    ai_prob = float(ai_prob.numpy()[0][0])
    ai_label = "AI_GENERATED" if ai_prob > 0.5 else "HUMAN"

    return ai_label, ai_prob
