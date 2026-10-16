import io
import os
import sys
import time
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))

from app.predictor import EchoGuardPredictor, TARGET_LEN, TARGET_SR

# Ensure UTF-8 stdout
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def get_gpu_memory_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    return allocated, reserved


def measure_latency_and_memory(predictor, test_audio_bytes, num_warmup=5, num_runs=30, use_fp16=False):
    is_cuda = predictor.device.type == "cuda"
    
    # Optional FP16 model conversion
    if use_fp16 and is_cuda:
        predictor.ssl_model.half()
        predictor.classifier.half()
    else:
        predictor.ssl_model.float()
        predictor.classifier.float()

    # Warmup
    for _ in range(num_warmup):
        _ = run_inference(predictor, test_audio_bytes, use_fp16=use_fp16)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    latencies_ms = []
    prep_times_ms = []
    backbone_times_ms = []
    head_times_ms = []

    for _ in range(num_runs):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = run_inference(predictor, test_audio_bytes, use_fp16=use_fp16)
        if is_cuda:
            torch.cuda.synchronize()
        t_total = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(t_total)
        prep_times_ms.append(res["latency"]["preprocess_ms"])
        backbone_times_ms.append(res["latency"]["backbone_ms"])
        head_times_ms.append(res["latency"]["classifier_ms"])

    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2) if is_cuda else 0.0
    peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2) if is_cuda else 0.0

    return {
        "mean_ms": np.mean(latencies_ms),
        "median_ms": np.median(latencies_ms),
        "p90_ms": np.percentile(latencies_ms, 90),
        "p99_ms": np.percentile(latencies_ms, 99),
        "min_ms": np.min(latencies_ms),
        "max_ms": np.max(latencies_ms),
        "prep_mean_ms": np.mean(prep_times_ms),
        "backbone_mean_ms": np.mean(backbone_times_ms),
        "head_mean_ms": np.mean(head_times_ms),
        "throughput_fps": 1000.0 / np.mean(latencies_ms),
        "peak_vram_alloc_mb": peak_alloc,
        "peak_vram_reserved_mb": peak_res,
        "sample_result": res,
    }


def run_inference(predictor, audio_bytes: bytes, use_fp16: bool = False):
    is_cuda = predictor.device.type == "cuda"
    
    # Ensure model weights match precision
    if use_fp16 and is_cuda:
        predictor.ssl_model.half()
        predictor.classifier.half()
    else:
        predictor.ssl_model.float()
        predictor.classifier.float()

    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        # Preprocess
        waveform = predictor.preprocess_audio(audio_bytes)
        if use_fp16 and is_cuda:
            waveform = waveform.half()
        else:
            waveform = waveform.float()

        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Backbone
        outputs = predictor.ssl_model(waveform)
        hidden = outputs.last_hidden_state
        mean_pool = hidden.mean(dim=1)
        std_pool = hidden.std(dim=1)
        embedding = torch.cat([mean_pool, std_pool], dim=1)

        if is_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Classifier
        if predictor.model_type == "oc_softmax":
            feats = predictor.classifier(embedding)
            score = float(predictor.criterion(feats)[0].item())
            bonafide_prob = float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
            decision_score = score
        else:
            logits = predictor.classifier(embedding)
            probs = torch.softmax(logits, dim=-1)[0]
            bonafide_prob = float(probs[1].item())
            decision_score = bonafide_prob

        if is_cuda:
            torch.cuda.synchronize()
        t3 = time.perf_counter()

    is_bonafide = bool(decision_score >= predictor.threshold)
    safe_score = max(decision_score, 1e-35)
    safe_thresh = max(predictor.threshold, 1e-35)
    margin_ratio = float(safe_score / safe_thresh)

    return {
        "prediction": "bonafide" if is_bonafide else "spoof",
        "bonafide_score": decision_score,
        "margin_ratio": margin_ratio,
        "latency": {
            "total_ms": round((t3 - t0) * 1000, 2),
            "preprocess_ms": round((t1 - t0) * 1000, 2),
            "backbone_ms": round((t2 - t1) * 1000, 2),
            "classifier_ms": round((t3 - t2) * 1000, 2),
        },
    }


def main():
    print("=" * 70)
    print("   ECHOGUARD PERFORMANCE, EFFICIENCY & ROBUSTNESS BENCHMARK")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware: {device.upper()} ({device_name})")

    # Load predictor
    checkpoint = "checkpoints/rawboost_ssl_best_model.pt"
    print(f"Loading checkpoint: {checkpoint}")
    predictor = EchoGuardPredictor(checkpoint_path=checkpoint, device=device)

    # Prepare benchmark audio samples
    # 1. Genuine speech
    bona_path = "release_in_the_wild/test/real/10018.wav"
    if not os.path.exists(bona_path):
        bona_path = "data/ASVspoof2019_LA_train/flac/LA_T_1138215.flac"
    with open(bona_path, "rb") as f:
        bona_bytes = f.read()

    # 2. AI Voice / Spoof
    ai_path = "data/test_ai_voice.wav"
    if not os.path.exists(ai_path):
        ai_path = "release_in_the_wild/test/fake/10012.wav"
    if not os.path.exists(ai_path):
        ai_path = "data/ASVspoof2019_LA_train/flac/LA_T_1000137.flac"
    with open(ai_path, "rb") as f:
        ai_bytes = f.read()

    # -------------------------------------------------------------
    # Test 1: FP32 Baseline Latency, Throughput & Memory
    # -------------------------------------------------------------
    print("\n[1/4] Running FP32 Baseline Benchmarks (30 runs)...")
    fp32_stats = measure_latency_and_memory(predictor, bona_bytes, use_fp16=False)
    print(f"  FP32 Mean Latency : {fp32_stats['mean_ms']:.2f} ms (P50: {fp32_stats['median_ms']:.2f} ms | P90: {fp32_stats['p90_ms']:.2f} ms | P99: {fp32_stats['p99_ms']:.2f} ms)")
    print(f"  FP32 Backbone Only: {fp32_stats['backbone_mean_ms']:.2f} ms")
    print(f"  FP32 Preprocess   : {fp32_stats['prep_mean_ms']:.2f} ms")
    print(f"  FP32 Head         : {fp32_stats['head_mean_ms']:.2f} ms")
    print(f"  FP32 Throughput   : {fp32_stats['throughput_fps']:.1f} requests/sec")
    print(f"  FP32 Peak VRAM    : {fp32_stats['peak_vram_alloc_mb']:.1f} MB allocated ({fp32_stats['peak_vram_reserved_mb']:.1f} MB reserved)")
    print(f"  FP32 Bonafide Score: {fp32_stats['sample_result']['bonafide_score']:.4e} -> {fp32_stats['sample_result']['prediction'].upper()}")

    # -------------------------------------------------------------
    # Test 2: FP16 Tensor-Core Optimization Latency, Throughput & Memory
    # -------------------------------------------------------------
    print("\n[2/4] Running FP16 Tensor-Core Optimization Benchmarks (30 runs)...")
    fp16_stats = measure_latency_and_memory(predictor, bona_bytes, use_fp16=True)
    speedup = fp32_stats["mean_ms"] / fp16_stats["mean_ms"]
    backbone_speedup = fp32_stats["backbone_mean_ms"] / fp16_stats["backbone_mean_ms"]
    vram_reduction = (1.0 - fp16_stats["peak_vram_alloc_mb"] / fp32_stats["peak_vram_alloc_mb"]) * 100.0

    print(f"  FP16 Mean Latency : {fp16_stats['mean_ms']:.2f} ms (P50: {fp16_stats['median_ms']:.2f} ms | P90: {fp16_stats['p90_ms']:.2f} ms | P99: {fp16_stats['p99_ms']:.2f} ms)")
    print(f"  FP16 Backbone Only: {fp16_stats['backbone_mean_ms']:.2f} ms (Speedup: {backbone_speedup:.2f}x)")
    print(f"  FP16 Throughput   : {fp16_stats['throughput_fps']:.1f} requests/sec (Speedup: {speedup:.2f}x)")
    print(f"  FP16 Peak VRAM    : {fp16_stats['peak_vram_alloc_mb']:.1f} MB allocated (Saved {vram_reduction:.1f}% VRAM)")
    print(f"  FP16 Bonafide Score: {fp16_stats['sample_result']['bonafide_score']:.4e} -> {fp16_stats['sample_result']['prediction'].upper()}")

    # -------------------------------------------------------------
    # Test 3: Numerical Decision Stability Verification (FP32 vs FP16)
    # -------------------------------------------------------------
    print("\n[3/4] Verifying Accuracy & Decision Stability (FP32 vs FP16)...")
    res_bona_fp32 = run_inference(predictor, bona_bytes, use_fp16=False)
    res_bona_fp16 = run_inference(predictor, bona_bytes, use_fp16=True)

    res_ai_fp32 = run_inference(predictor, ai_bytes, use_fp16=False)
    res_ai_fp16 = run_inference(predictor, ai_bytes, use_fp16=True)

    print("  Genuine Human Sample:")
    print(f"    FP32: score = {res_bona_fp32['bonafide_score']:.4e} | verdict = {res_bona_fp32['prediction'].upper()}")
    print(f"    FP16: score = {res_bona_fp16['bonafide_score']:.4e} | verdict = {res_bona_fp16['prediction'].upper()}")
    assert res_bona_fp32['prediction'] == res_bona_fp16['prediction'] == "bonafide", "Decision divergence on human sample!"

    print("  AI Voice Sample:")
    print(f"    FP32: score = {res_ai_fp32['bonafide_score']:.4e} | verdict = {res_ai_fp32['prediction'].upper()}")
    print(f"    FP16: score = {res_ai_fp16['bonafide_score']:.4e} | verdict = {res_ai_fp16['prediction'].upper()}")
    assert res_ai_fp32['prediction'] == res_ai_fp16['prediction'] == "spoof", "Decision divergence on AI sample!"
    print("  -> Precision check passed: 100% agreement between FP32 and FP16 decisions!")

    # -------------------------------------------------------------
    # Test 4: Robustness Stress Test (Noise, Attenuation & Clipping)
    # -------------------------------------------------------------
    print("\n[4/4] Robustness Stress Testing (Noise, Low Gain, Heavy Gain)...")
    d, sr = sf.read(io.BytesIO(bona_bytes))
    if d.ndim > 1:
        d = d.mean(axis=1)

    # 4a. Additive Gaussian noise (SNR ~ 15dB)
    noise = np.random.randn(len(d)).astype(np.float32)
    signal_power = np.mean(d ** 2)
    noise_power = signal_power / (10 ** (15 / 10))
    d_noisy = d + noise * np.sqrt(noise_power)
    bio_noisy = io.BytesIO()
    sf.write(bio_noisy, d_noisy, sr, format="WAV")
    res_noisy = run_inference(predictor, bio_noisy.getvalue(), use_fp16=True)
    print(f"  [15 dB SNR Noise]  Score: {res_noisy['bonafide_score']:.4e} -> Verdict: {res_noisy['prediction'].upper()} (Target: BONAFIDE)")

    # 4b. Low gain (0.05x amplitude - quiet laptop mic)
    d_quiet = d * 0.05
    bio_quiet = io.BytesIO()
    sf.write(bio_quiet, d_quiet, sr, format="WAV")
    res_quiet = run_inference(predictor, bio_quiet.getvalue(), use_fp16=True)
    print(f"  [0.05x Low Gain]   Score: {res_quiet['bonafide_score']:.4e} -> Verdict: {res_quiet['prediction'].upper()} (Target: BONAFIDE)")

    # 4c. AI voice with noise (ensure spoof detection survives noise)
    d_ai, sr_ai = sf.read(io.BytesIO(ai_bytes))
    if d_ai.ndim > 1:
        d_ai = d_ai.mean(axis=1)
    noise_ai = np.random.randn(len(d_ai)).astype(np.float32)
    sig_p_ai = np.mean(d_ai ** 2)
    noise_p_ai = sig_p_ai / (10 ** (15 / 10))
    d_ai_noisy = d_ai + noise_ai * np.sqrt(noise_p_ai)
    bio_ai_noisy = io.BytesIO()
    sf.write(bio_ai_noisy, d_ai_noisy, sr_ai, format="WAV")
    res_ai_noisy = run_inference(predictor, bio_ai_noisy.getvalue(), use_fp16=True)
    print(f"  [AI Voice + Noise] Score: {res_ai_noisy['bonafide_score']:.4e} -> Verdict: {res_ai_noisy['prediction'].upper()} (Target: SPOOF)")

    print("\n" + "=" * 70)
    print("                  BENCHMARK SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} | {'FP32 Baseline':<18} | {'FP16 Optimized':<18} | {'Improvement':<12}")
    print("-" * 75)
    print(f"{'Mean Latency':<25} | {fp32_stats['mean_ms']:>14.2f} ms | {fp16_stats['mean_ms']:>14.2f} ms | {speedup:>10.2f}x")
    print(f"{'Backbone Latency':<25} | {fp32_stats['backbone_mean_ms']:>14.2f} ms | {fp16_stats['backbone_mean_ms']:>14.2f} ms | {backbone_speedup:>10.2f}x")
    print(f"{'P99 Latency':<25} | {fp32_stats['p99_ms']:>14.2f} ms | {fp16_stats['p99_ms']:>14.2f} ms | {fp32_stats['p99_ms']/fp16_stats['p99_ms']:>10.2f}x")
    print(f"{'Throughput':<25} | {fp32_stats['throughput_fps']:>11.1f} req/s | {fp16_stats['throughput_fps']:>11.1f} req/s | {speedup:>10.2f}x")
    print(f"{'Peak VRAM (Allocated)':<25} | {fp32_stats['peak_vram_alloc_mb']:>15.1f} MB | {fp16_stats['peak_vram_alloc_mb']:>15.1f} MB | -{vram_reduction:>8.1f}%")
    print(f"{'Decision Accuracy':<25} | {'100% Matching':>18} | {'100% Matching':>18} | {'Exact':>12}")
    print("=" * 75)


if __name__ == "__main__":
    main()
