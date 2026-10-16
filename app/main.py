import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure app and models are in python path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

from predictor import EchoGuardPredictor

predictor: EchoGuardPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("[EchoGuard API] Starting up and loading models...")
    checkpoint = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "rawboost_ssl_best_model.pt")
    predictor = EchoGuardPredictor(checkpoint_path=checkpoint, use_fp16=True)
    yield
    print("[EchoGuard API] Shutting down...")


app = FastAPI(
    title="EchoGuard Voice Spoof Detection API",
    description="Real-time anti-spoofing and deepfake voice detection using wav2vec2-base SSL embeddings and MLP classifier.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def get_dashboard():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "EchoGuard API is running. Dashboard available at /static/index.html"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "EchoGuard Voice Spoof Detection",
        "model": "Model B (wav2vec2-base + SSLClassifierMLP with RawBoost)",
        "device": str(predictor.device) if predictor else "uninitialized",
        "precision": "FP16 (Tensor-Core Accelerated)" if (predictor and predictor.use_fp16) else "FP32",
        "checkpoint": os.path.basename(predictor.classifier.__class__.__name__) if predictor else "none",
        "in_the_wild_eer": "26.01%",
        "asvspoof19_eval_eer": "3.54%",
        "operating_threshold": predictor.threshold if predictor else None,
        "inference_latency": "~15 ms (FP16)",
    }


@app.post("/score")
async def score_audio(file: UploadFile = File(...)):
    """
    Classify an uploaded audio recording as bonafide (human) or spoof (deepfake/synthetic/cloned).
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model inference engine is not ready.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    allowed_exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")
    if not any(file.filename.lower().endswith(ext) for ext in allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Please upload one of: {', '.join(allowed_exts)}",
        )

    try:
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = predictor.predict(audio_bytes)
        result["filename"] = file.filename
        result["file_size_kb"] = round(len(audio_bytes) / 1024, 2)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/api/sample/{sample_type}")
def get_sample_audio(sample_type: str):
    """
    Returns sample audio from the dataset for quick testing.
    sample_type: 'bonafide' or 'spoof'
    """
    if sample_type == "bonafide":
        wild_path = os.path.join(os.path.dirname(__file__), "..", "release_in_the_wild", "test", "real", "10018.wav")
        if os.path.exists(wild_path):
            sample_path = wild_path
        else:
            sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "ASVspoof2019_LA_train", "flac", "LA_T_1138215.flac")
    elif sample_type == "spoof":
        wild_path = os.path.join(os.path.dirname(__file__), "..", "release_in_the_wild", "test", "fake", "10012.wav")
        if os.path.exists(wild_path):
            sample_path = wild_path
        else:
            sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "ASVspoof2019_LA_train", "flac", "LA_T_1000137.flac")
    else:
        raise HTTPException(status_code=404, detail="Unknown sample type. Use 'bonafide' or 'spoof'.")

    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample audio file not found on disk.")

    media_type = "audio/wav" if sample_path.endswith(".wav") else "audio/flac"
    return FileResponse(sample_path, media_type=media_type, filename=os.path.basename(sample_path))
