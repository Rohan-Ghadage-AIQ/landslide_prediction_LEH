# 🏔️ Leh-Ladakh Landslide Early Warning & Hazard Prediction System

A machine-learning-powered geospatial system designed to predict landslide risk levels across the high-altitude terrain of Leh-Ladakh using Earth Observation (EO) satellite data and Geological Survey of India (GSI) historical landslide records.

---

## 📌 1. Project Overview

Landslides and debris flows pose severe risks to infrastructure, military transit corridors, and local communities across the Himalayan region of Leh-Ladakh. This project integrates multi-spectral satellite remote sensing, digital elevation models, climate rainfall datasets, and machine learning to evaluate real-time landslide susceptibility.

---

## 🛰️ 2. Data Sources: What We Are Using, Why, & Where It Comes From

To accurately predict landslide hazards, the system combines **topographic**, **environmental**, **meteorological**, and **historical ground-truth** data.

| Data Feature | Why We Use It (Physical Rationale) | Data Source & Provider | Resolution / Format |
| :--- | :--- | :--- | :--- |
| **Slope (Degrees)** | **Gravitational Instability**: Steep slopes ($>25^\circ-30^\circ$) exponentially increase shear stress on rock and soil. | **USGS / NASA SRTM DEM** *(US Geological Survey)* | 30m Spatial Resolution |
| **Elevation (Meters)** | **Freeze-Thaw & Permafrost**: In high-altitude Ladakh ($>3,500m$), freeze-thaw cycles break down rock faces, causing rockfalls. | **USGS / NASA SRTM DEM** *(US Geological Survey)* | 30m Spatial Resolution |
| **Aspect (Degrees)** | **Solar & Thermal Exposure**: South-facing slopes receive higher solar radiation, accelerating snowmelt and ground moisture changes. | **USGS / NASA SRTM DEM** *(US Geological Survey)* | 30m Spatial Resolution |
| **NDVI (Vegetation Index)** | **Root Cohesion & Anchoring**: Vegetation roots bind soil together. Low NDVI ($<0.2$) indicates bare rock, loose scree, or unanchored soil. | **ESA Sentinel-2 Satellite** *(European Space Agency)* | 10m Multi-Spectral Imagery |
| **Rainfall (mm)** | **Hydrological Trigger**: Heavy rainfall or cloudburst events saturate soil, reducing shear strength and triggering mass movement. | **CHIRPS Climate Dataset** *(UCSB Climate Hazards Center)* | Daily Gridded Rainfall |
| **Ground-Truth Landslides** | **Model Training & Calibration**: Verified historical landslide and debris flow incident coordinates across the Leh region. | **GSI NLSM & ISRO Bhuvan** *(Geological Survey of India)* | Spatial Georeferenced Coordinates |

---

## 🏗️ 3. System Architecture & Workflow

```
┌────────────────────────────────┐
│  Google Earth Engine Extractor │  ---> Downloads SRTM DEM, Sentinel-2 NDVI, & CHIRPS Rainfall GeoTIFFs
│       (gee_extractor.py)       │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│      Geospatial Data Pipeline  │  ---> Overlays GSI Landslide Coordinates onto Rasters
│       (data_pipeline.py)       │       Generates balanced training set (training_data.csv)
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│     Machine Learning Model     │  ---> Trains Random Forest Classifier
│      (model_training.py)       │       Exports model binary (landslide_model.pkl)
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│       FastAPI REST Service     │  ---> Serves live risk predictions & status endpoints
│           (main.py)            │       `POST /api/predict`
└────────────────────────────────┘
```

---

## 🤖 4. Machine Learning Model & Risk Classification

- **Algorithm**: `RandomForestClassifier` (100 Decision Trees, Max Depth 10)
- **Features Used for Prediction**:
  1. `elevation` (m)
  2. `slope` (degrees)
  3. `aspect` (degrees)
  4. `ndvi` (-1.0 to 1.0)
  5. `rainfall_mm` (mm)
  6. `soil_type` (Categorical)

### Risk Classification Thresholds
The model calculates a continuous probability score ($0.0$ to $1.0$) for landslide occurrence:
- 🟢 **Low Risk**: Probability $< 0.30$ (Flat terrain, dense vegetation, low rainfall)
- 🟡 **Moderate Risk**: Probability between $0.30$ and $0.70$ (Moderate slopes or partial vegetation)
- 🔴 **High Risk**: Probability $> 0.70$ (Steep slopes, bare scree/soil, heavy precipitation)

---

## 🚀 5. How to Run & Quickstart Guide

### Prerequisites
Make sure Python 3.10+ is installed along with project dependencies:
```bash
pip install -r requirements.txt
```

### Step 1: Run Data Pipeline & Train Model
```bash
# 1. Process GSI Landslide coordinates & satellite rasters into training data
python -c "from data_pipeline import generate_training_data_from_gsi_points; generate_training_data_from_gsi_points('data/gsi_leh_landslides.csv')"

# 2. Train the Random Forest model
python model_training.py
```

### Step 2: Launch the FastAPI Backend Server
```bash
python main.py
```
*(Server will start at `http://127.0.0.1:8000`)*

---

## 🧪 6. Testing the API

### Check API & Model Status
```bash
curl http://127.0.0.1:8000/api/status
```
**Response**:
```json
{
  "status": "ok",
  "message": "Backend is running.",
  "model_loaded": true
}
```

### Request a Landslide Risk Prediction
```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "elevation": 4500.0,
       "slope": 65.0,
       "aspect": 180.0,
       "ndvi": 0.05,
       "rainfall_mm": 120.0,
       "soil_type": 2
     }'
```
**Output**:
```json
{
  "landslide_probability": 0.92,
  "risk_level": "High"
}
```

### Interactive API Documentation
While `main.py` is running, open **`http://127.0.0.1:8000/docs`** in your browser to test endpoints interactively via Swagger UI.

---

## 📄 License & Attribution
- Satellite datasets provided courtesy of **USGS/NASA** (SRTM), **ESA** (Sentinel-2), and **UCSB CHIRPS**.
- Landslide inventory points sourced from **Geological Survey of India (GSI) NLSM**.
