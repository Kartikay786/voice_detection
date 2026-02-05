import base64
import io
import numpy as np
import soundfile as sf
import librosa

def decode_base64_to_audio(b64_string):
    # ---- FIX INCORRECT PADDING ----
    missing_padding = len(b64_string) % 4
    if missing_padding:
        b64_string += "=" * (4 - missing_padding)

    audio_bytes = base64.b64decode(b64_string)
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    return audio, sr

def extract_melspectrogram(audio, sr=16000, target_len=157):
    """
    Memory-safe Mel extraction for long audio.
    Always returns shape (128, target_len).
    """

    # ---- 1) Convert to mono if stereo ----
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # ---- 2) Resample to 16 kHz (prevents huge arrays) ----
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # ---- 3) Limit audio to max 10 seconds ----
    max_len = sr * 10        # 10 seconds
    if len(audio) > max_len:
        audio = audio[:max_len]

    # ---- 4) Ensure at least 1 second ----
    if len(audio) < sr:
        audio = np.pad(audio, (0, sr - len(audio)))

    # ---- 5) Compute Mel spectrogram ----
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # ---- 6) Force exact width for your CNN ----
    if mel_db.shape[1] < target_len:
        mel_db = np.pad(mel_db, ((0, 0), (0, target_len - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :target_len]

    return mel_db
