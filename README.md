<p align="center">
  <h1 align="center">🛡️ EchoGuard — Audio Deepfake Detection System</h1>
  <p align="center">
    <em>Real-time AI-generated voice detection using Self-Supervised Learning (wav2vec2) with RawBoost augmentation and FP16 Tensor-Core acceleration</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/wav2vec2-HuggingFace-yellow?logo=huggingface&logoColor=white" alt="HuggingFace">
    <img src="https://img.shields.io/badge/CUDA-FP16_TensorCore-76B900?logo=nvidia&logoColor=white" alt="CUDA">
  </p>
</p>

---

## 📌 Overview

**EchoGuard** is an end-to-end audio deepfake detection system that identifies AI-generated speech (from TTS engines like ElevenLabs, Bark, VALL-E, etc.) vs genuine human voice recordings. It combines a frozen **facebook/wav2vec2-base** SSL backbone with a lightweight MLP classifier, trained using **RawBoost** data augmentation and evaluated across both controlled benchmarks and real-world "in-the-wild" audio.

### Key Highlights

- 🧠 **SSL-Based Architecture** — Frozen wav2vec2-base (768-dim) with mean+std pooling → 1536-dim embeddings
- 🔊 **RawBoost Augmentation** — 50/50 clean + augmented training mixture for noise robustness
- ⚡ **FP16 Tensor-Core Inference** — ~15ms latency, 2.13× speedup, 45.9% VRAM savings on RTX 2050
- 🌐 **Real-Time Web Interface** — Beautiful glassmorphic UI with drag-and-drop upload + live microphone recording
- 🔬 **2×2 Factorial Ablation** — Systematic study of RawBoost × OC-Softmax combinations
- 📊 **Cross-Dataset Generalization** — Evaluated on both ASVspoof 2019 LA and In-the-Wild benchmarks

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EchoGuard Pipeline                       │
│                                                                 │
│  Raw Audio (16kHz) ──► Peak Normalize ──► Speech Onset Detect   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────┐                            │
│  │  wav2vec2-base (Frozen, FP16)   │  ← 95M params, no grad    │
│  │  12 Transformer Layers          │                            │
│  └────────────┬────────────────────┘                            │
│               │                                                 │
│               ▼                                                 │
│  ┌─────────────────────────────────┐                            │
│  │  Mean + Std Pooling → 1536-dim  │                            │
│  └────────────┬────────────────────┘                            │
│               │                                                 │
│               ▼                                                 │
│  ┌─────────────────────────────────┐                            │
│  │  MLP Classifier (256 hidden)    │  ← Trained from scratch    │
│  │  Dropout 0.3 + Softmax          │                            │
│  └────────────┬────────────────────┘                            │
│               │                                                 │
│               ▼                                                 │
│     bonafide / spoof (with calibrated threshold)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results

### Model Evolution (4-Model Ablation Study)

| Model | Architecture | Training Strategy | Loss | ASVspoof 2019 LA EER | In-the-Wild EER |
|:---|:---|:---|:---|:---:|:---:|
| **Phase 1 Baseline** | Log-Mel + 5-Layer CNN | Clean | CE | 6.16% | — |
| **Model A** | wav2vec2 + MLP | Clean | CE | **2.72%** | 26.94% |
| **Model B** ⭐ | wav2vec2 + MLP | 50% Clean + 50% RawBoost | CE | 3.54% | **26.01%** |
| **Model C** | wav2vec2 + OC-Proj | Clean | OC-Softmax | 3.15% | 31.19% |
| **Model D** | wav2vec2 + OC-Proj | 50% Clean + 50% RawBoost | OC-Softmax | 3.49% | 34.38% |

> **Best Model (Model B):** RawBoost + Cross-Entropy Softmax achieves the best cross-dataset generalization.

### FP16 Efficiency Benchmarks (RTX 2050, 4GB VRAM)

| Metric | FP32 | FP16 | Improvement |
|:---|:---:|:---:|:---:|
| Inference Latency | ~32ms | **~15ms** | **2.13× faster** |
| VRAM Usage | ~1.8GB | **~1.0GB** | **45.9% savings** |
| Decision Parity | — | **100%** | No accuracy loss |

---

## 🗂️ Project Structure

```
EchoGuard/
├── app/                          # FastAPI web application
│   ├── main.py                   # API endpoints (/score, /health)
│   ├── predictor.py              # Inference engine (FP16, calibrated threshold)
│   └── static/                   # Frontend UI
│       ├── index.html            # Glassmorphic web interface
│       ├── style.css             # Premium dark-theme styles
│       └── app.js                # Upload, recording, and analysis logic
├── data/                         # Dataset utilities (data NOT included)
│   ├── dataset.py                # ASVspoof dataset loader (mel spectrograms)
│   ├── embedding_dataset.py      # Pre-extracted embedding loader
│   ├── protocol_parser.py        # ASVspoof protocol file parser
│   ├── explore.py                # Dataset statistics & exploration
│   ├── RawBoost.py               # RawBoost augmentation algorithms
│   └── rawboost_pipeline.py      # Augmentation pipeline (50/50 mixture)
├── models/                       # Neural network definitions
│   ├── cnn_baseline.py           # Phase 1: Log-Mel CNN baseline
│   ├── ssl_mlp.py                # Phase 2: MLP classifier head
│   └── oc_softmax.py             # Phase 2: OC-Softmax loss + classifier
├── scripts/                      # Utility & benchmarking scripts
│   └── benchmark_efficiency.py   # FP16 vs FP32 latency/VRAM benchmark
├── reports/                      # Evaluation results & analysis
│   └── results.md                # Full benchmark results table
├── train.py                      # Phase 1: CNN baseline training
├── train_ssl.py                  # Phase 2: SSL-MLP clean training
├── train_ssl_rawboost.py         # Phase 2: SSL-MLP + RawBoost training
├── train_ssl_ocsoftmax.py        # Phase 2: SSL + OC-Softmax training
├── extract_embeddings.py         # Extract wav2vec2 embeddings (clean)
├── extract_embeddings_rawboost.py # Extract embeddings (RawBoost augmented)
├── calibrate_threshold.py        # Calibrate operating threshold from EER
├── eval_in_the_wild.py           # Cross-dataset evaluation script
├── evaluate.py                   # EER computation utility
├── test_eval.py                  # ASVspoof eval set evaluation
├── test_eval_ssl.py              # SSL model evaluation
├── test_api.py                   # FastAPI integration tests
├── run_server.py                 # Server entry point
├── download_model.py             # Download pre-trained weights from GitHub Releases
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git exclusions (data, checkpoints, venv)
```

---

## 🗃️ Datasets (Download Separately)

> ⚠️ **Datasets are NOT included in this repository due to their large size.** Download them from the official sources below.

### 1. ASVspoof 2019 LA (Primary Training & Evaluation)

| Split | Samples | Usage |
|:---|:---:|:---|
| Train | 25,380 | Model training |
| Dev | 24,844 | Validation & early stopping |
| Eval | 71,237 | Final EER evaluation |

**Download:** [ASVspoof 2019 on Kaggle](https://www.kaggle.com/datasets/awsaf49/asvspoof-2019-dataset) or [Official ASVspoof Site](https://www.asvspoof.org/index2019.html)

After downloading, place the files so the structure matches:
```
data/
├── ASVspoof2019_LA_train/flac/       # Training .flac files
├── ASVspoof2019_LA_dev/flac/         # Development .flac files
├── ASVspoof2019_LA_eval/flac/        # Evaluation .flac files
└── ASVspoof2019_LA_cm_protocols/     # Protocol .txt files
```

### 2. In-the-Wild Dataset (Cross-Dataset Generalization)

Real-world audio from YouTube, podcasts, and various TTS engines (3,179 utterances).

**Download:** [In-the-Wild Dataset](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) by Müller et al. (2022)

Place the files under:
```
release_in_the_wild/
├── train/
├── val/
├── test/
└── attribution.txt
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/<your-username>/EchoGuard.git
cd EchoGuard
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Download Pre-trained Model (Recommended — No Training Needed)

```bash
python download_model.py
```

This downloads the pre-trained model weights (~5 MB) from GitHub Releases into `checkpoints/`. You can skip straight to running the app — **no GPU, no dataset, no training required.**

### 3. Run the Web Application

```bash
python run_server.py
```

Open **http://localhost:8000** in your browser. Upload an audio file or use the microphone to record and analyze.

### 4. (Optional) Retrain from Scratch

Only needed if you want to reproduce the full training pipeline:

```bash
# Download datasets first (see Dataset section above)

# Step 1: Extract wav2vec2 embeddings from clean audio
python extract_embeddings.py

# Step 2: Extract RawBoost-augmented embeddings (50/50 mixture)
python extract_embeddings_rawboost.py

# Step 3: Train the MLP classifier on combined embeddings
python train_ssl_rawboost.py

# Step 4: Calibrate the operating threshold
python calibrate_threshold.py

# Step 5: Evaluate on ASVspoof 2019 LA eval set
python test_eval_ssl.py

# Step 6: Cross-dataset evaluation on In-the-Wild
python eval_in_the_wild.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|:---|:---|
| **SSL Backbone** | facebook/wav2vec2-base (HuggingFace Transformers) |
| **Framework** | PyTorch 2.0+ with CUDA FP16 |
| **Web Backend** | FastAPI + Uvicorn |
| **Web Frontend** | Vanilla HTML/CSS/JS (Glassmorphic UI) |
| **Audio Processing** | torchaudio, soundfile |
| **Augmentation** | RawBoost (Tak et al., 2022) |
| **Loss Functions** | Cross-Entropy, OC-Softmax (Zhang et al., 2021) |
| **Hardware** | Optimized for NVIDIA RTX 2050 (4GB VRAM) |

---

## 📚 Key Findings

1. **RawBoost 50/50 Mixture > 100% Augmentation** — Training on 100% corrupted audio starved the model of clean reference, degrading EER from 2.72% to 7.98%. The 50/50 mixture restored performance and improved generalization.

2. **Cross-Entropy Softmax > OC-Softmax for Generalization** — While OC-Softmax excels in-domain (3.15% EER), standard softmax with RawBoost delivers superior open-set generalization (26.01% vs 34.38% In-the-Wild EER).

3. **FP16 = Free Performance** — Half-precision inference provides 2.13× speedup and 45.9% VRAM savings with zero accuracy loss (100% decision parity).

---

## 📖 References

- **ASVspoof 2019:** Todisco, M., et al. "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection." *Interspeech*, 2019.
- **RawBoost:** Tak, H., et al. "RawBoost: A Raw Data Boosting and Augmentation Method Applied to Automatic Speaker Verification Anti-Spoofing." *ICASSP*, 2022.
- **OC-Softmax:** Zhang, Y., et al. "One-Class Learning Towards Synthetic Voice Spoofing Detection." *IEEE Signal Processing Letters*, 2021.
- **wav2vec 2.0:** Baevski, A., et al. "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." *NeurIPS*, 2020.
- **In-the-Wild Dataset:** Müller, N.M., et al. "Does Audio Deepfake Detection Generalize?" *Interspeech*, 2022.

---

## ⚙️ Hardware Requirements

| Component | Minimum | Recommended |
|:---|:---|:---|
| **GPU** | NVIDIA GTX 1650 (4GB) | NVIDIA RTX 2050+ (4GB+) |
| **RAM** | 8GB | 16GB |
| **CPU** | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 |
| **Storage** | 10GB (code + checkpoints) | 50GB (with datasets) |
| **CUDA** | 11.8+ | 12.0+ |

> 💡 CPU-only inference is supported but runs at ~200ms latency vs ~15ms on GPU.

---

## 📄 License

This project is for educational and research purposes. The ASVspoof and In-the-Wild datasets have their own licensing terms — refer to their official sources.

---

<p align="center">
  <strong>Built with ❤️ for deepfake audio research</strong>
</p>
