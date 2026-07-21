import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AOI_SHP_PATH = os.path.join(BASE_DIR, "Leh_Ladakh_AOI", "leh_ladakh.shp")
MODEL_PATH = os.path.join(BASE_DIR, "landslide_model.pkl")

# Model configuration
FEATURES = ["elevation", "slope", "aspect", "ndvi", "rainfall_mm", "soil_type"]
TARGET = "landslide_occurred"

# Dummy configurations for spatial grid if we generate mock data
GRID_RESOLUTION = 0.01 # Approx 1km at equator
