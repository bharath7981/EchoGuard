import io
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Model

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from ssl_mlp import SSLClassifierMLP
from oc_softmax import OCSoftmaxLoss, SSLClassifierOC

TARGET_SR = 16000
TARGET_LEN = 64600  # ~4.04s at 16kHz


def pad_or_crop(waveform: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    current_len = len(waveform)
    if current_len == target_len:
        return waveform
    if current_len > target_len:
        return waveform[:target_len]
    repeat_count = target_len // current_len + 1
    waveform = np.tile(waveform, repeat_count)
    return waveform[:target_len]


# Calibrated operating threshold for Model B on In-the-Wild benchmark (EER = 26.01%)
DEFAULT_CALIBRATED_THRESHOLD = 1.5179394408093572e-09


class EchoGuardPredictor:
    """
    Production-grade inference engine for EchoGuard spoof detection.
    Preloads facebook/wav2vec2-base and the trained SSLClassifierMLP into GPU memory.
    Uses calibrated decision threshold derived from In-the-Wild benchmark.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/rawboost_ssl_best_model.pt",
        threshold: float = DEFAULT_CALIBRATED_THRESHOLD,
        device: str = None,
        use_fp16: bool = True,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.threshold = float(threshold)
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")
        precision_str = "FP16 (Tensor-Core Accelerated)" if self.use_fp16 else "FP32 (Single Precision)"
        print(f"[EchoGuardPredictor] Initializing on device: {self.device} ({torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'})")
        print(f"[EchoGuardPredictor] Precision: {precision_str}")
        print(f"[EchoGuardPredictor] Operating threshold: {self.threshold:.10e}")

        # 1. Load wav2vec2-base feature extractor
        print("[EchoGuardPredictor] Loading facebook/wav2vec2-base backbone...")
        self.ssl_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
        self.ssl_model.eval()
        for param in self.ssl_model.parameters():
            param.requires_grad = False

        # 2. Load trained classifier
        print(f"[EchoGuardPredictor] Loading classifier weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model_type = checkpoint.get("model_type", "standard_mlp")

        if self.model_type == "oc_softmax":
            print("[EchoGuardPredictor] Detected OC-Softmax model architecture.")
            hidden_dim = checkpoint.get("hidden_dim", 256)
            self.classifier = SSLClassifierOC(in_features=1536, hidden_dim=hidden_dim).to(self.device)
            self.classifier.load_state_dict(checkpoint["model_state_dict"])
            self.classifier.eval()

            self.criterion = OCSoftmaxLoss(in_features=hidden_dim).to(self.device)
            self.criterion.load_state_dict(checkpoint["criterion_state_dict"])
            self.criterion.eval()
        else:
            print("[EchoGuardPredictor] Detected standard Softmax MLP model architecture.")
            self.classifier = SSLClassifierMLP(in_features=1536, hidden_dim=256, num_classes=2).to(self.device)
            self.classifier.load_state_dict(checkpoint["model_state_dict"])
            self.classifier.eval()
            self.criterion = None

        for param in self.classifier.parameters():
            param.requires_grad = False

        # Convert to FP16 half-precision for ~2x throughput and 45% VRAM savings
        if self.use_fp16:
            self.ssl_model = self.ssl_model.half()
            self.classifier = self.classifier.half()

        # 3. GPU Warmup to eliminate initial cold-start kernel compilation latency
        if self.device.type == "cuda":
            print("[EchoGuardPredictor] Warming up GPU kernels...")
            dtype = torch.float16 if self.use_fp16 else torch.float32
            dummy = torch.zeros((1, TARGET_LEN), dtype=dtype, device=self.device)
            with torch.inference_mode():
                out = self.ssl_model(dummy).last_hidden_state
                emb = torch.cat([out.mean(dim=1), out.std(dim=1)], dim=1)
                _ = self.classifier(emb)
            torch.cuda.synchronize()

        print("[EchoGuardPredictor] Inference engine ready.")

    def preprocess_audio(self, audio_bytes: bytes) -> torch.Tensor:
        """
        Decodes bytes, converts to mono 16kHz, selects active speech window,
        and normalizes dynamic range to match standard studio/benchmark audio.
        """
        bio = io.BytesIO(audio_bytes)
        try:
            data, sr = sf.read(bio, dtype="float32")
        except Exception as err:
            raise ValueError(
                f"Audio format not recognized or unsupported container ({err}). "
                "Please upload or record standard WAV, FLAC, OGG, or MP3 audio."
            )

        # Mono mix
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to 16kHz
        if sr != TARGET_SR:
            import torchaudio.functional as F
            tensor_data = torch.from_numpy(data).unsqueeze(0)
            resampled = F.resample(tensor_data, sr, TARGET_SR)
            data = resampled.squeeze(0).numpy()

        # If audio is longer than 4.04s, trim initial silence to speech onset
        if len(data) > TARGET_LEN:
            max_amp = float(np.max(np.abs(data)))
            if max_amp > 1e-4:
                speech_indices = np.where(np.abs(data) > 0.15 * max_amp)[0]
                if len(speech_indices) > 0:
                    start_idx = max(0, speech_indices[0] - 1600)  # keep 100ms lead-in
                    if len(data) - start_idx >= TARGET_LEN:
                        data = data[start_idx : start_idx + TARGET_LEN]
                    else:
                        data = data[start_idx:]

        # Peak normalize to standard dynamic range (0.90 peak) to eliminate mic gain sensitivity
        max_val = float(np.max(np.abs(data)))
        if max_val > 1e-4:
            data = (data / max_val) * 0.90

        data = pad_or_crop(data, TARGET_LEN)
        return torch.from_numpy(data).unsqueeze(0).to(self.device)  # (1, 64600)

    def predict(self, audio_bytes: bytes) -> dict:
        """
        Takes raw audio bytes and returns spoof/bonafide classification,
        calibrated decision margin relative to operating threshold, and stage-by-stage latency.
        Uses Tensor-Core FP16 half-precision on GPU when enabled.
        """
        is_cuda = self.device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            # 1. Preprocess
            waveform = self.preprocess_audio(audio_bytes)
            if self.use_fp16 and is_cuda:
                waveform = waveform.half()

            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # 2. Extract SSL embedding (Mean + Std pooling)
            outputs = self.ssl_model(waveform)
            hidden = outputs.last_hidden_state  # (1, T, 768)

            mean_pool = hidden.mean(dim=1)      # (1, 768)
            std_pool = hidden.std(dim=1)        # (1, 768)
            embedding = torch.cat([mean_pool, std_pool], dim=1)  # (1, 1536)
            if is_cuda:
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            # 3. Classify with MLP or OC-Softmax
            if self.model_type == "oc_softmax":
                feats = self.classifier(embedding)
                score = float(self.criterion(feats)[0].item())  # cosine similarity in [-1, 1]
                bonafide_prob = float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
                spoof_prob = 1.0 - bonafide_prob
                decision_score = score
            else:
                logits = self.classifier(embedding)
                probs = torch.softmax(logits, dim=-1)[0]
                spoof_prob = float(probs[0].item())
                bonafide_prob = float(probs[1].item())
                decision_score = bonafide_prob

            if is_cuda:
                torch.cuda.synchronize()
            t3 = time.perf_counter()

        # Calibrated decision against operating threshold
        is_bonafide = bool(decision_score >= self.threshold)
        prediction = "bonafide" if is_bonafide else "spoof"

        # Ratio and log10 margin relative to decision threshold
        safe_score = max(decision_score, 1e-35)
        safe_thresh = max(self.threshold, 1e-35)
        margin_ratio = float(safe_score / safe_thresh)
        log10_score = float(np.log10(safe_score))
        log10_thresh = float(np.log10(safe_thresh))
        log_margin = float(log10_score - log10_thresh)

        preprocess_ms = round((t1 - t0) * 1000, 1)
        backbone_ms = round((t2 - t1) * 1000, 1)
        classifier_ms = round((t3 - t2) * 1000, 1)
        total_latency_ms = round((t3 - t0) * 1000, 1)

        margin_desc = f"{margin_ratio:,.0f}× above threshold" if is_bonafide else f"{margin_ratio:.9e}× of threshold"

        return {
            "prediction": prediction,
            "is_bonafide": is_bonafide,
            "bonafide_score": decision_score,
            "spoof_score": spoof_prob if self.model_type != "oc_softmax" else 1.0 - decision_score,
            "threshold": self.threshold,
            "margin_ratio": margin_ratio,
            "log_margin": round(log_margin, 2),
            "margin_desc": margin_desc,
            "latency": {
                "total_ms": total_latency_ms,
                "preprocess_ms": preprocess_ms,
                "backbone_ms": backbone_ms,
                "classifier_ms": classifier_ms,
            },
            "latency_ms": total_latency_ms,
            "device": str(self.device),
            "precision": "FP16 (Tensor-Core)" if self.use_fp16 else "FP32",
            "model_type": self.model_type,
            # Legacy fields for backward compatibility
            "bonafide_probability": float(decision_score),
            "spoof_probability": float(spoof_prob if self.model_type != "oc_softmax" else 1.0 - decision_score),
            "confidence": round(float(np.clip(0.5 + 0.5 * min(abs(log_margin) / 5.0, 1.0), 0.5, 1.0)), 4),
        }


if __name__ == "__main__":
    predictor = EchoGuardPredictor()
    test_path = "data/ASVspoof2019_LA_train/flac/LA_T_1138215.flac"
    if os.path.exists(test_path):
        with open(test_path, "rb") as f:
            res = predictor.predict(f.read())
        print(f"Test prediction for {os.path.basename(test_path)} (known bonafide):", res)
