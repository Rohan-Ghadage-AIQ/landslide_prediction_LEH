import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse

import config
from data_pipeline import generate_real_data_for_leh
from model_training import train_model

app = FastAPI(title="Leh Landslide Early Warning Backend", version="1.0")

# Enable CORS for the future dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_ui():
    html_path = os.path.join(config.BASE_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Leh Landslide Early Warning Backend is running. index.html not found."}

class PredictRequest(BaseModel):
    elevation: float
    slope: float
    aspect: float
    ndvi: float
    rainfall_mm: float
    soil_type: int

class PredictResponse(BaseModel):
    landslide_probability: float
    risk_level: str

@app.get("/api/status")
def get_status():
    model_exists = os.path.exists(config.MODEL_PATH)
    return {
        "status": "ok",
        "message": "Backend is running.",
        "model_loaded": model_exists
    }

@app.post("/api/train")
def run_training_pipeline(samples: int = 2000):
    """
    Triggers the generation of mock data from the Leh shapefile and trains the ML model.
    """
    try:
        # Step 1: Generate Data
        df = generate_real_data_for_leh(samples)
        
        # Step 2: Train Model
        metrics = train_model()
        
        if metrics is None:
             raise HTTPException(status_code=500, detail="Model training failed.")
             
        return {
            "status": "success",
            "message": "Data pipeline and model training completed successfully.",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict", response_model=PredictResponse)
def predict_risk(request: PredictRequest):
    """
    Predicts the landslide probability for a given set of parameters.
    """
    if not os.path.exists(config.MODEL_PATH):
        raise HTTPException(status_code=400, detail="Model is not trained yet. Call /api/train first.")
        
    try:
        model = joblib.load(config.MODEL_PATH)
        
        # Create DataFrame for prediction matching training feature order
        df = pd.DataFrame([{
            "elevation": request.elevation,
            "slope": request.slope,
            "aspect": request.aspect,
            "ndvi": request.ndvi,
            "rainfall_mm": request.rainfall_mm,
            "soil_type": request.soil_type
        }])
        
        # Predict probability of class 1 (landslide occurred)
        probability = float(model.predict_proba(df)[0][1])
        
        # Assign risk level based on probability
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Moderate"
        else:
            risk_level = "High"
            
        return PredictResponse(
            landslide_probability=probability,
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
