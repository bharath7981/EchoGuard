# EchoGuard Audio Deepfake Detection - Research Results & Benchmarks

## 1. Evolution of Model Benchmarks (Updated with Corrected 50/50 Pipeline)

| Model Configuration | Architecture | Training Strategy | Loss Function | ASVspoof 2019 LA Eval EER | In-the-Wild EER (Primary Metric) | In-the-Wild FAR (at 0.5) | In-the-Wild Miss Rate (at 0.5) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 1 Baseline** | Log-Mel Spectrogram + 5-Layer CNN | Supervised (Clean) | Cross-Entropy | **6.16%** | — | — | — |
| **Model A (Clean SSL Baseline)** | wav2vec2-base (Frozen) + MLP | Clean 100% Studio | Cross-Entropy | **2.72%** | **26.94%** | **94.94%** | **0.34%** |
| **Model B (+RawBoost 50/50 Mixture)** | wav2vec2-base (Frozen) + MLP | 50% Clean + 50% RawBoost | Cross-Entropy | **3.54%** | **`26.01%`** *(Best)* | **97.50%** | **0.17%** |
| **Model C (+OC-Softmax Only)** | wav2vec2-base (Frozen) + OC Projection | Clean 100% Studio | OC-Softmax ($\alpha=20$) | **3.15%** | **31.19%** | **91.94%** | **0.51%** |
| **Model D (+RawBoost 50/50 + OC-Softmax)** | wav2vec2-base (Frozen) + OC Projection | 50% Clean + 50% RawBoost | OC-Softmax ($\alpha=20$) | **3.49%** | **34.38%** | **89.08%** | **0.93%** |

---

## 2. The 2×2 Factorial Ablation Matrix (Ranked by In-the-Wild EER)

This $2 \times 2$ factorial matrix isolates the effect of Data Augmentation (**RawBoost**) and Metric Learning (**One-Class Softmax**) using threshold-independent Equal Error Rate (EER):

| Loss Function \ Data Pipeline | Baseline (Clean Studio Data) | + RawBoost (50/50 Clean + Algo 4 Mixture) |
| :--- | :--- | :--- |
| **Standard Cross-Entropy Softmax** | **Model A (Clean Baseline)**<br>• In-the-Wild EER: **26.94%**<br>• ASVspoof19 Eval EER: **2.72%** | **Model B (+RawBoost)**<br>• In-the-Wild EER: **`26.01%`** *(▲ **0.93%** absolute improvement)*<br>• ASVspoof19 Eval EER: **3.54%** |
| **One-Class Softmax (OC-Softmax)** | **Model C (+OC-Softmax Control)**<br>• In-the-Wild EER: **31.19%**<br>• ASVspoof19 Eval EER: **3.15%** | **Model D (+RawBoost + OC-Softmax)**<br>• In-the-Wild EER: **34.38%**<br>• ASVspoof19 Eval EER: **3.49%** |

---

## 3. Scientific Findings & Root Cause Analysis

1. **Fixing the Augmentation Pipeline (100% Corrupted vs 50/50 Mixture):**
   - In our initial trial, training on 100% statically corrupted RawBoost data starved the model of pristine reference audio, causing in-domain EER to degrade 3× (2.72% $\to$ 7.98%) and out-of-domain EER to balloon to 35.70%.
   - By constructing a **50/50 mixture (25,380 clean + 25,380 augmented = 50,760 samples)**, the model retained clean acoustic reference while acquiring noise invariance.
   - **Result:** In-domain EER was restored from 7.98% down to **3.54%**, and In-the-Wild EER dropped from 35.70% down to **26.01%** (officially beating the 26.94% clean baseline).

2. **The Impact of Smoothed Early Stopping:**
   - Raw dev EER on noisy tasks exhibits high variance. Moving to a 3-epoch moving average with patience 10 allowed the models to train for 32 epochs (Model B) and 26 epochs (Model D), reaching genuine convergence rather than terminating prematurely at epoch 12.

3. **Metric Learning vs Standard Softmax Out-of-Domain:**
   - While OC-Softmax achieves strong in-domain performance (3.15% clean, 3.49% augmented), standard Cross-Entropy Softmax with RawBoost (Model B) delivers the superior open-set generalization on diverse real-world audio (**26.01% vs 34.38%**).
