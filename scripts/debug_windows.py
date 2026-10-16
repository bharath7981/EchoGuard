import io
import soundfile as sf
import numpy as np
import torch
import torchaudio.functional as F
import httpx

client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0)

for fpath in ["data/test_ai_voice.wav", "data/test_ai_voice_2.wav"]:
    print(f"\nAnalyzing: {fpath}")
    d, sr = sf.read(fpath)
    if d.ndim > 1:
        d = d.mean(axis=1)
    if sr != 16000:
        d = F.resample(torch.from_numpy(d).unsqueeze(0), sr, 16000).squeeze(0).numpy()

    hop = 16000 // 2
    for start in range(0, max(1, len(d) - 64600 + 1), hop):
        chunk = d[start : start + 64600]
        if len(chunk) < 64600:
            chunk = np.pad(chunk, (0, 64600 - len(chunk)))
        chunk = chunk / (np.max(np.abs(chunk)) + 1e-6) * 0.9
        bio = io.BytesIO()
        sf.write(bio, chunk, 16000, format="WAV", subtype="PCM_16")
        bio.seek(0)
        r = client.post("/score", files={"file": ("chunk.wav", bio.read(), "audio/wav")}).json()
        b_score = r.get("bonafide_score", 0.0)
        pred = r.get("prediction", "")
        print(f"  Window {start/16000:.1f}s - {(start+64600)/16000:.1f}s: bonafide_score={b_score:.4e} -> {pred}")
