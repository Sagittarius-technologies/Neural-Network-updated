import sys
import os
import uuid
import json
import time
import shutil
import zipfile
import traceback
import io
from pathlib import Path
from typing import Dict, Any

# Add the backend directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pipeline  # Direct, reliable import

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent
RUNS_DIR = ROOT_DIR / "runs"
MODELS_DIR = RUNS_DIR / "models"
PREDICTIONS_DIR = RUNS_DIR / "predictions"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="eDNA NN API", version="1.0.0")

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
    """Save uploaded file to destination path."""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return True
    except Exception as e:
        print(f"Error saving file {destination}: {e}")
        return False
    finally:
        upload_file.file.close()

def write_job_status(run_dir: Path, status_data: Dict[str, Any]):
    """Write job status to JSON file."""
    try:
        status_file = run_dir / "status.json"
        with status_file.open("w") as f:
            json.dump(status_data, f, indent=4)
    except Exception as e:
        print(f"Error writing status file: {e}")

def validate_model_files(model_dir: Path) -> bool:
    """Validate that all required model files exist."""
    required_files = [
        "kmer_vectorizer.joblib",
        "reducer.joblib", 
        "scaler.joblib",
        "label_encoder.joblib",
        "mlp_model.joblib"
    ]
    
    for file_name in required_files:
        if not (model_dir / file_name).exists():
            print(f"Missing required model file: {file_name}")
            return False
    return True

# --- Background Task Logic ---
def run_training_task(run_id: str, fasta_path_str: str, params: Dict[str, Any]):
    """Background task for model training."""
    model_dir = MODELS_DIR / run_id
    fasta_path = Path(fasta_path_str)
    
    # Initialize status
    status_data = {
        "run_id": run_id, 
        "status": "running", 
        "start_time": time.time(),
        "end_time": None, 
        "error": None,
        "parameters": {**params, "filename": fasta_path.name}, 
        "job_type": "training"
    }
    write_job_status(model_dir, status_data)
    
    print(f"[{run_id}] Starting training with parameters: {params}")
    
    try:
        # Validate input file
        if not fasta_path.exists():
            raise FileNotFoundError(f"Training file not found: {fasta_path}")
        
        if fasta_path.stat().st_size == 0:
            raise ValueError("Training file is empty")
        
        # Call training function
        model_path = pipeline.train_from_labeled_fasta(
            train_fasta=str(fasta_path),
            outdir=str(model_dir),
            k=int(params["k"]),
            pca_comp=int(params["pca_comp"]),
            epochs=int(params["epochs"]),
            centroid_percentile=float(params["centroid_percentile"])
        )
        
        # Validate output files
        if not validate_model_files(model_dir):
            raise RuntimeError("Training completed but some model files are missing")
        
        status_data["status"] = "completed"
        status_data["model_path"] = model_path
        print(f"[{run_id}] Training completed successfully.")
        
    except Exception as e:
        error_str = str(e)
        traceback_str = traceback.format_exc()
        status_data.update({"status": "failed", "error": error_str, "traceback": traceback_str})
        print(f"[{run_id}] Training failed: {error_str}\nTraceback:\n{traceback_str}")
        
    finally:
        status_data["end_time"] = time.time()
        write_job_status(model_dir, status_data)
        
        # Cleanup input file if it's temporary
        try:
            if fasta_path.exists() and fasta_path.name.startswith("uploaded"):
                fasta_path.unlink()
        except:
            pass

def run_prediction_task(run_id: str, model_run_id: str, fasta_path_str: str, params: Dict[str, Any]):
    """Background task for prediction."""
    prediction_dir = PREDICTIONS_DIR / run_id
    model_dir = MODELS_DIR / model_run_id
    fasta_path = Path(fasta_path_str)
    
    # Initialize status
    status_data = {
        "run_id": run_id, 
        "status": "running", 
        "start_time": time.time(),
        "end_time": None, 
        "error": None,
        "parameters": {**params, "model_run_id": model_run_id, "filename": fasta_path.name}, 
        "job_type": "prediction"
    }
    write_job_status(prediction_dir, status_data)
    
    print(f"[{run_id}] Starting prediction using model {model_run_id} with parameters: {params}")
    
    try:
        # Validate input file
        if not fasta_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {fasta_path}")
        
        if fasta_path.stat().st_size == 0:
            raise ValueError("Prediction file is empty")
        
        # Validate model files
        if not validate_model_files(model_dir):
            raise RuntimeError(f"Model directory {model_run_id} is missing required files")
        
        # Call prediction function with corrected parameters
        assign_df, medoid_df, labels = pipeline.cluster_and_predict(
            raw_fasta=str(fasta_path),
            model_dir=str(model_dir),  # Pass model directory instead of individual paths
            outdir=str(prediction_dir),
            k=int(params.get("k", 4)),
            pca_comp=int(params.get("pca_comp", 50)),
            cluster_method=str(params.get("cluster_method", "kmeans")),
            dbscan_eps=float(params.get("dbscan_eps", 0.5)),
            dbscan_min=int(params.get("dbscan_min", 5)),
            kmeans_n=int(params.get("kmeans_n", 10)),
            threshold=float(params.get("threshold", 0.7))
        )
        
        # Validate output files
        expected_outputs = ["cluster_medoid_predictions.csv", "sequence_cluster_assignments.csv", "species_abundance_by_reads.csv"]
        missing_outputs = []
        for output_file in expected_outputs:
            if not (prediction_dir / output_file).exists():
                missing_outputs.append(output_file)
        
        if missing_outputs:
            print(f"Warning: Some expected output files are missing: {missing_outputs}")
        
        status_data["status"] = "completed"
        status_data["results_info"] = {
            "total_sequences": len(assign_df) if assign_df is not None else 0,
            "num_clusters": len(set(labels)) - (1 if -1 in labels else 0) if labels is not None else 0,
            "medoids_found": len(medoid_df) if medoid_df is not None else 0
        }
        print(f"[{run_id}] Prediction completed successfully.")
        
    except Exception as e:
        error_str = str(e)
        traceback_str = traceback.format_exc()
        status_data.update({"status": "failed", "error": error_str, "traceback": traceback_str})
        print(f"[{run_id}] Prediction failed: {error_str}\nTraceback:\n{traceback_str}")
        
    finally:
        status_data["end_time"] = time.time()
        write_job_status(prediction_dir, status_data)
        
        # Cleanup input file if it's temporary
        try:
            if fasta_path.exists() and fasta_path.name.startswith("uploaded"):
                fasta_path.unlink()
        except:
            pass

# --- API Endpoints ---
@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the eDNA NN API", 
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "train": "POST /train",
            "predict": "POST /predict", 
            "status": "GET /runs/{run_id}",
            "models": "GET /models",
            "download": "GET /runs/{run_id}/download"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/train")
async def train_model(
    background_tasks: BackgroundTasks, 
    fasta: UploadFile = File(...), 
    k: int = Form(4, description="K-mer size"),
    pca_comp: int = Form(50, description="PCA components"),
    epochs: int = Form(50, description="Training epochs"),
    centroid_percentile: float = Form(95.0, description="Centroid distance percentile")
):
    """Start model training with labeled FASTA file."""
    try:
        # Validate file
        if not fasta.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not fasta.filename.lower().endswith(('.fasta', '.fa', '.fas')):
            raise HTTPException(status_code=400, detail="File must be a FASTA file (.fasta, .fa, .fas)")
        
        # Validate parameters
        if k < 1 or k > 10:
            raise HTTPException(status_code=400, detail="K-mer size must be between 1 and 10")
        if pca_comp < 1 or pca_comp > 1000:
            raise HTTPException(status_code=400, detail="PCA components must be between 1 and 1000")
        if epochs < 1 or epochs > 1000:
            raise HTTPException(status_code=400, detail="Epochs must be between 1 and 1000")
        if centroid_percentile < 50 or centroid_percentile > 99.9:
            raise HTTPException(status_code=400, detail="Centroid percentile must be between 50 and 99.9")
        
        # Generate run ID and create directory
        run_id = f"train_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        model_dir = MODELS_DIR / run_id
        model_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        fasta_path = model_dir / fasta.filename
        if not save_uploaded_file(fasta, fasta_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        
        # Prepare parameters
        params = {
            "k": k, 
            "pca_comp": pca_comp, 
            "epochs": epochs, 
            "centroid_percentile": centroid_percentile
        }
        
        # Start background task
        background_tasks.add_task(run_training_task, run_id, str(fasta_path), params)
        
        return {
            "message": "Training job started successfully", 
            "run_id": run_id,
            "parameters": params,
            "status_endpoint": f"/runs/{run_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Training endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks, 
    raw_fasta: UploadFile = File(...), 
    model_run_id: str = Form(..., description="Model run ID to use for prediction"),
    cluster_method: str = Form("kmeans", description="Clustering method: kmeans or dbscan"),
    kmeans_n: int = Form(10, description="Number of clusters for K-means"),
    dbscan_eps: float = Form(0.5, description="DBSCAN epsilon parameter"),
    dbscan_min: int = Form(5, description="DBSCAN min samples parameter"),
    threshold: float = Form(0.7, description="Confidence threshold for predictions")
):
    """Start prediction using trained model."""
    try:
        # Validate model exists
        model_dir = MODELS_DIR / model_run_id
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model with run_id '{model_run_id}' not found")
        
        # Check model status
        status_file = model_dir / "status.json"
        if status_file.exists():
            with status_file.open("r") as f:
                model_status = json.load(f)
            if model_status.get("status") != "completed":
                raise HTTPException(status_code=400, detail="Model training is not completed yet")
        
        # Validate model files
        if not validate_model_files(model_dir):
            raise HTTPException(status_code=400, detail="Model files are incomplete or missing")
        
        # Validate input file
        if not raw_fasta.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not raw_fasta.filename.lower().endswith(('.fasta', '.fa', '.fas')):
            raise HTTPException(status_code=400, detail="File must be a FASTA file (.fasta, .fa, .fas)")
        
        # Validate parameters
        if cluster_method.lower() not in ["kmeans", "dbscan"]:
            raise HTTPException(status_code=400, detail="Cluster method must be 'kmeans' or 'dbscan'")
        if kmeans_n < 1 or kmeans_n > 100:
            raise HTTPException(status_code=400, detail="K-means clusters must be between 1 and 100")
        if dbscan_eps <= 0 or dbscan_eps > 10:
            raise HTTPException(status_code=400, detail="DBSCAN eps must be between 0 and 10")
        if dbscan_min < 1 or dbscan_min > 50:
            raise HTTPException(status_code=400, detail="DBSCAN min samples must be between 1 and 50")
        if threshold < 0 or threshold > 1:
            raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
        
        # Generate run ID and create directory
        run_id = f"predict_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        prediction_dir = PREDICTIONS_DIR / run_id
        prediction_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        fasta_path = prediction_dir / raw_fasta.filename
        if not save_uploaded_file(raw_fasta, fasta_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        
        # Prepare parameters
        params = {
            "cluster_method": cluster_method,
            "kmeans_n": kmeans_n,
            "dbscan_eps": dbscan_eps,
            "dbscan_min": dbscan_min,
            "threshold": threshold
        }
        
        # Start background task
        background_tasks.add_task(run_prediction_task, run_id, model_run_id, str(fasta_path), params)
        
        return {
            "message": "Prediction job started successfully", 
            "run_id": run_id,
            "model_run_id": model_run_id,
            "parameters": params,
            "status_endpoint": f"/runs/{run_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/runs/{run_id}")
def get_run_status(run_id: str):
    """Get status and results of a training or prediction run."""
    try:
        # Determine run directory
        if run_id.startswith("predict"):
            run_dir = PREDICTIONS_DIR / run_id
        elif run_id.startswith("train"):
            run_dir = MODELS_DIR / run_id
        else:
            raise HTTPException(status_code=400, detail="Invalid run_id format")
        
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail="Run ID not found")
        
        # Read status file
        status_file = run_dir / "status.json"
        if not status_file.exists():
            raise HTTPException(status_code=404, detail="Status file for this run not found")
        
        with status_file.open("r") as f:
            job_details = json.load(f)
        
        # Add runtime information
        if job_details.get("start_time") and job_details.get("end_time"):
            job_details["runtime_seconds"] = job_details["end_time"] - job_details["start_time"]
        elif job_details.get("start_time"):
            job_details["current_runtime_seconds"] = time.time() - job_details["start_time"]
        
        # Prepare results summary
        results = {}
        if job_details["status"] == "completed":
            if run_id.startswith("predict"):
                # Get prediction results summary
                try:
                    abundance_file = run_dir / "species_abundance_by_reads.csv"
                    if abundance_file.exists():
                        import pandas as pd
                        df = pd.read_csv(abundance_file)
                        results["species_summary"] = {
                            "total_species": len(df),
                            "top_species": df.head(5).to_dict('records') if len(df) > 0 else []
                        }
                except Exception as e:
                    print(f"Error reading abundance file: {e}")
                    results["species_summary"] = {"error": "Could not read results"}
            
            results["download_available"] = True
        
        return {
            "job_details": job_details, 
            "results": results,
            "download_endpoint": f"/runs/{run_id}/download" if job_details["status"] == "completed" else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Status endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/models")
def get_models():
    """Get list of completed trained models."""
    try:
        completed_models = []
        
        for model_dir in MODELS_DIR.iterdir():
            if not model_dir.is_dir():
                continue
                
            status_file = model_dir / "status.json"
            if not status_file.exists():
                continue
                
            try:
                with status_file.open("r") as f:
                    status_data = json.load(f)
                
                if status_data.get("status") == "completed":
                    end_time = status_data.get("end_time")
                    formatted_time = "N/A"
                    
                    if isinstance(end_time, (int, float)):
                        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
                    
                    runtime = None
                    if status_data.get("start_time") and end_time:
                        runtime = end_time - status_data["start_time"]
                    
                    completed_models.append({
                        "run_id": status_data["run_id"],
                        "end_time": formatted_time,
                        "runtime_seconds": runtime,
                        "parameters": status_data.get("parameters", {}),
                        "files_valid": validate_model_files(model_dir)
                    })
            except Exception as e:
                print(f"Error reading model status {model_dir.name}: {e}")
                continue
        
        # Sort by end time (most recent first)
        completed_models.sort(key=lambda x: x.get("end_time", ""), reverse=True)
        
        return {
            "models": completed_models,
            "total_count": len(completed_models)
        }
        
    except Exception as e:
        print(f"Models endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/runs/{run_id}/download")
def download_run_files(run_id: str):
    """Download all files from a completed run as a ZIP archive."""
    try:
        # Determine run directory
        if run_id.startswith("predict"):
            run_dir = PREDICTIONS_DIR / run_id
        elif run_id.startswith("train"):
            run_dir = MODELS_DIR / run_id
        else:
            raise HTTPException(status_code=400, detail="Invalid run_id format")
        
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail="Run ID not found")
        
        # Check if run is completed
        status_file = run_dir / "status.json"
        if status_file.exists():
            with status_file.open("r") as f:
                status_data = json.load(f)
            if status_data.get("status") != "completed":
                raise HTTPException(status_code=400, detail="Run is not completed yet")
        
        # Create ZIP file in memory
        mem_zip = io.BytesIO()
        
        with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            file_count = 0
            for f_path in run_dir.glob('*'):
                if f_path.is_file() and f_path.name != "status.json":  # Exclude status file
                    try:
                        zf.write(f_path, f_path.name)
                        file_count += 1
                    except Exception as e:
                        print(f"Error adding file {f_path.name} to zip: {e}")
            
            if file_count == 0:
                raise HTTPException(status_code=404, detail="No files found to download")
        
        mem_zip.seek(0)
        
        return StreamingResponse(
            io.BytesIO(mem_zip.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={run_id}_results.zip"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Download endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# For debugging - remove in production
@app.get("/debug/runs")
def debug_runs():
    """Debug endpoint to show all run directories."""
    return {
        "models_dir": str(MODELS_DIR),
        "predictions_dir": str(PREDICTIONS_DIR),
        "model_runs": [d.name for d in MODELS_DIR.iterdir() if d.is_dir()],
        "prediction_runs": [d.name for d in PREDICTIONS_DIR.iterdir() if d.is_dir()]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
