import base64
import io
import numpy as np
import soundfile as sf
import librosa

def decode_base64_to_audio(b64_string):
    audio_bytes = base64.b64decode(b64_string)
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    return audio, sr

def extract_melspectrogram(audio, sr=16000):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db
