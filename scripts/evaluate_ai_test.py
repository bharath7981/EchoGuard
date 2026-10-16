import sys
import httpx

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0)

tests = [
    ("Generated AI Voice 1 (Standard TTS)", "data/test_ai_voice.wav"),
    ("Generated AI Voice 2 (Fast TTS)", "data/test_ai_voice_2.wav"),
]

print("\n" + "=" * 65)
print("  ECHOGUARD AI VOICE EVALUATION TEST REPORT")
print("=" * 65)

for label, path in tests:
    with open(path, "rb") as f:
        res = client.post("/score", files={"file": (path, f, "audio/wav")}).json()
    pred = res.get("prediction", "").upper()
    verdict_str = "🔴 AI VOICE DETECTED" if pred == "SPOOF" else "🟢 HUMAN VOICE"
    b_score = res.get("bonafide_score", 0.0)
    thresh = res.get("threshold", 0.0)
    margin = res.get("margin_desc", "")
    log_m = res.get("log_margin", 0.0)
    lat = res.get("latency", {}).get("total_ms", 0.0)

    print(f"\n[TEST CASE] {label}")
    print(f"  Verdict:          {verdict_str}")
    print(f"  Prediction Key:   {pred}")
    print(f"  Bonafide Score:   {b_score:.4e}")
    print(f"  Threshold:        {thresh:.4e}")
    print(f"  Separation:       {margin} (log10 margin: {log_m:+.2f})")
    print(f"  Latency (GPU):    {lat:.1f} ms")
    print(f"  Accuracy Status:  {'CORRECT (AI voice detected)' if pred == 'SPOOF' else 'INCORRECT'}")

# Score the genuine benchmark as well
r_bona = client.get("/api/sample/bonafide")
res_bona = client.post("/score", files={"file": ("benchmark_human.flac", r_bona.content, "audio/flac")}).json()
pred_b = res_bona.get("prediction", "").upper()

print(f"\n[BASELINE BENCHMARK] Genuine Human Speech")
print(f"  Verdict:          {'🟢 HUMAN VOICE' if pred_b == 'BONAFIDE' else '🔴 AI VOICE DETECTED'}")
print(f"  Prediction Key:   {pred_b}")
print(f"  Bonafide Score:   {res_bona.get('bonafide_score', 0.0):.4e}")
print(f"  Threshold:        {res_bona.get('threshold', 0.0):.4e}")
print(f"  Separation:       {res_bona.get('margin_desc', '')}")
print(f"  Latency (GPU):    {res_bona.get('latency', {}).get('total_ms', 0.0):.1f} ms")
print(f"  Accuracy Status:  {'CORRECT (Human voice detected)' if pred_b == 'BONAFIDE' else 'INCORRECT'}")

print("\n" + "=" * 65 + "\n")
