import sys
import os
import uuid
import json
import time
import shutil
import zipfile
import traceback
import io  # <-- FIX: Added missing import
from pathlib import Path
from typing import Dict, Any

# Add the backend directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import backend.pipeline_old as pipeline_old  # Direct, reliable import

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent
RUNS_DIR = ROOT_DIR / "runs"
MODELS_DIR = RUNS_DIR / "models"
PREDICTIONS_DIR = RUNS_DIR / "predictions"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="eDNA NN API")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
def save_uploaded_file(upload_file: UploadFile, destination: Path):
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

def write_job_status(run_dir: Path, status_data: Dict[str, Any]):
    status_file = run_dir / "status.json"
    with status_file.open("w") as f:
        json.dump(status_data, f, indent=4)

# --- Background Task Logic ---
def run_training_task(run_id: str, fasta_path_str: str, params: Dict[str, Any]):
    model_dir = MODELS_DIR / run_id
    fasta_path = Path(fasta_path_str)
    status_data = {
        "run_id": run_id, "status": "running", "start_time": time.time(),
        "end_time": None, "error": None,
        "parameters": {**params, "filename": fasta_path.name}, "job_type": "training"
    }
    write_job_status(model_dir, status_data)
    print(f"[{run_id}] Starting training...")
    try:
        pipeline_old.train_from_labeled_fasta(
            train_fasta=str(fasta_path),
            outdir=str(model_dir),
            k=int(params["k"]),
            pca_comp=int(params["pca_comp"]),
            epochs=int(params["epochs"]),
            centroid_percentile=float(params["centroid_percentile"])
        )
        status_data["status"] = "completed"
        print(f"[{run_id}] Training completed successfully.")
    except Exception as e:
        error_str = str(e)
        traceback_str = traceback.format_exc()
        status_data.update({"status": "failed", "error": error_str})
        print(f"[{run_id}] !!! TRAINING FAILED !!!\nError: {error_str}\nTraceback:\n{traceback_str}")
    finally:
        status_data["end_time"] = time.time()
        write_job_status(model_dir, status_data)

def run_prediction_task(run_id: str, model_run_id: str, fasta_path_str: str, params: Dict[str, Any]):
    prediction_dir = PREDICTIONS_DIR / run_id
    model_dir = MODELS_DIR / model_run_id
    fasta_path = Path(fasta_path_str)
    status_data = {
        "run_id": run_id, "status": "running", "start_time": time.time(),
        "end_time": None, "error": None,
        "parameters": {**params, "model_run_id": model_run_id, "filename": fasta_path.name}, "job_type": "prediction"
    }
    write_job_status(prediction_dir, status_data)
    print(f"[{run_id}] Starting prediction using model {model_run_id}...")
    try:
        vec_path = model_dir / "kmer_vectorizer.joblib"
        reducer_path = model_dir / "reducer.joblib"
        scaler_path = model_dir / "scaler.joblib"
        encoder_path = model_dir / "label_encoder.joblib"
        mlp_model_path = model_dir / "mlp_model.joblib"
        
        pipeline_old.cluster_and_predict(
            raw_fasta=str(fasta_path),
            model_path=str(mlp_model_path),
            vec_path=str(vec_path),
            reducer_path=str(reducer_path),
            scaler_path=str(scaler_path),
            encoder_path=str(encoder_path),
            outdir=str(prediction_dir),
            cluster_method=str(params["cluster_method"]),
            kmeans_n=int(params["kmeans_n"]),
            threshold=float(params["threshold"])
        )
        status_data["status"] = "completed"
        print(f"[{run_id}] Prediction completed successfully.")
    except Exception as e:
        error_str = str(e)
        traceback_str = traceback.format_exc()
        status_data.update({"status": "failed", "error": error_str})
        print(f"[{run_id}] !!! PREDICTION FAILED !!!\nError: {error_str}\nTraceback:\n{traceback_str}")
    finally:
        status_data["end_time"] = time.time()
        write_job_status(prediction_dir, status_data)

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the eDNA NN API. Navigate to /docs for the interactive API documentation."}

@app.post("/train")
async def train_model(
    background_tasks: BackgroundTasks, fasta: UploadFile = File(...), k: int = Form(4),
    pca_comp: int = Form(50), epochs: int = Form(50), centroid_percentile: float = Form(95.0),
):
    run_id = f"train_{uuid.uuid4()}"
    model_dir = MODELS_DIR / run_id
    model_dir.mkdir(exist_ok=True)
    fasta_path = model_dir / (fasta.filename or "uploaded.fasta")
    save_uploaded_file(fasta, fasta_path)
    params = {"k": k, "pca_comp": pca_comp, "epochs": epochs, "centroid_percentile": centroid_percentile}
    background_tasks.add_task(run_training_task, run_id, str(fasta_path), params)
    return {"message": "Training job started successfully.", "run_id": run_id}

@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks, raw_fasta: UploadFile = File(...), model_run_id: str = Form(...),
    cluster_method: str = Form("kmeans"), kmeans_n: int = Form(10), threshold: float = Form(0.7),
):
    if not (MODELS_DIR / model_run_id).exists():
        raise HTTPException(status_code=404, detail=f"Model with run_id '{model_run_id}' not found.")
    run_id = f"predict_{uuid.uuid4()}"
    prediction_dir = PREDICTIONS_DIR / run_id
    prediction_dir.mkdir(exist_ok=True)
    fasta_path = prediction_dir / (raw_fasta.filename or "uploaded.fasta")
    save_uploaded_file(raw_fasta, fasta_path)
    params = {"cluster_method": cluster_method, "kmeans_n": kmeans_n, "threshold": threshold}
    background_tasks.add_task(run_prediction_task, run_id, model_run_id, str(fasta_path), params)
    return {"message": "Prediction job started successfully.", "run_id": run_id}

@app.get("/runs/{run_id}")
def get_run_status(run_id: str):
    run_dir = PREDICTIONS_DIR / run_id if run_id.startswith("predict") else MODELS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run ID not found.")
    status_file = run_dir / "status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Status file for this run not found.")
    with status_file.open("r") as f:
        job_details = json.load(f)
    results = {}
    if job_details["status"] == "completed":
        if run_id.startswith("predict"):
            try:
                report_path = run_dir / "species_abundance_by_reads.csv"
                if report_path.exists():
                     results = {"summary": "Prediction complete. Download files to view results."}
            except Exception:
                results = {"summary": "Prediction complete, but report could not be automatically generated."}
    return {"job_details": job_details, "results": results}

@app.get("/models")
def get_models():
    completed_models = []
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir():
            status_file = model_dir / "status.json"
            if status_file.exists():
                with status_file.open("r") as f:
                    status_data = json.load(f)
                if status_data.get("status") == "completed":
                    # FIX: Handle cases where end_time might be None or not a number
                    end_time = status_data.get("end_time")
                    formatted_time = "N/A"
                    if isinstance(end_time, (int, float)):
                        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
                    
                    completed_models.append({
                        "run_id": status_data["run_id"],
                        "end_time": formatted_time,
                        "parameters": status_data.get("parameters", {})
                    })
    completed_models.sort(key=lambda x: x["end_time"], reverse=True)
    return completed_models

@app.get("/runs/{run_id}/download")
def download_run_files(run_id: str):
    run_dir = PREDICTIONS_DIR / run_id if run_id.startswith("predict") else MODELS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run ID not found.")
    
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f_path in run_dir.glob('*'):
            if f_path.is_file():
                zf.write(f_path, f_path.name)
    mem_zip.seek(0)
    return StreamingResponse(
        mem_zip, media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={run_id}_results.zip"}
    )

