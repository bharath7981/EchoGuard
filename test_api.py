import os
import sys
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(__file__))
from app.main import app

def run_tests():
    print("Initializing TestClient and loading models...")
    with TestClient(app) as client:
        # 1. Test /health
        print("\n[1/3] Testing GET /health...")
        res = client.get("/health")
        assert res.status_code == 200, f"Health check failed: {res.status_code}"
        health_data = res.json()
        print("Health Status:", health_data)
        assert health_data["status"] == "healthy"

        # 2. Test /score on known bonafide
        print("\n[2/3] Testing POST /score with known bonafide sample...")
        bonafide_path = "data/ASVspoof2019_LA_train/flac/LA_T_1138215.flac"
        assert os.path.exists(bonafide_path), f"Sample file {bonafide_path} not found"
        
        with open(bonafide_path, "rb") as f:
            res = client.post("/score", files={"file": ("LA_T_1138215.flac", f, "audio/flac")})
        assert res.status_code == 200, f"/score failed: {res.text}"
        score_bonafide = res.json()
        print("Bonafide Sample Result:", score_bonafide)
        assert score_bonafide["prediction"] == "bonafide"

        # 3. Test /score on known spoof (AI deepfake)
        print("\n[3/3] Testing POST /score with known spoof sample...")
        spoof_path = "data/test_ai_voice.wav"
        if not os.path.exists(spoof_path):
            spoof_path = "release_in_the_wild/test/fake/10012.wav"
        assert os.path.exists(spoof_path), f"Sample file {spoof_path} not found"
        
        with open(spoof_path, "rb") as f:
            res = client.post("/score", files={"file": (os.path.basename(spoof_path), f, "audio/wav")})
        assert res.status_code == 200, f"/score failed: {res.text}"
        score_spoof = res.json()
        print("Spoof Sample Result:", score_spoof)
        assert score_spoof["prediction"] == "spoof"

        print("\nAll automated API tests passed successfully! [PASS]")

if __name__ == "__main__":
    run_tests()
