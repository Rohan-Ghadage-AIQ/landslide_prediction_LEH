import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
import config

def get_pixel_value(lon, lat, tif_files):
    """
    Given a list of tif files, finds the one that contains the (lon, lat)
    and returns its pixel value(s). Returns None if not found.
    """
    for tif in tif_files:
        with rasterio.open(tif) as src:
            # Check if point is within the bounding box of this tile
            bounds = src.bounds
            if bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top:
                # Get the pixel row, col
                row, col = src.index(lon, lat)
                
                # Check if it's within the actual array size (sometimes index can be just outside due to floating point)
                if 0 <= row < src.height and 0 <= col < src.width:
                    # Read all bands for this pixel
                    # generator of values, we unpack it
                    val = list(src.sample([(lon, lat)]))
                    if val and len(val) > 0:
                        return val[0] # Returns a numpy array of band values
    return None

def generate_real_data_for_leh(num_samples: int = 2000):
    """
    Generates training data by sampling random points within the Leh shapefile.
    Extracts real features for elevation, slope, aspect, ndvi, rainfall from GEE GeoTIFFs.
    """
    print(f"Loading shapefile from {config.AOI_SHP_PATH}...")
    if not os.path.exists(config.AOI_SHP_PATH):
        raise FileNotFoundError(f"Shapefile not found: {config.AOI_SHP_PATH}")
    
    aoi_gdf = gpd.read_file(config.AOI_SHP_PATH)
    leh_geom = aoi_gdf.geometry.iloc[0]
    minx, miny, maxx, maxy = leh_geom.bounds
    
    # Locate Data Files
    data_dir = os.path.join(config.BASE_DIR, "data")
    terrain_files = glob.glob(os.path.join(data_dir, "**", "Leh_Terrain_DEM_Slope_Aspect*.tif"), recursive=True)
    ndvi_files = glob.glob(os.path.join(data_dir, "**", "Leh_Sentinel2_NDVI*.tif"), recursive=True)
    rainfall_files = glob.glob(os.path.join(data_dir, "**", "Leh_CHIRPS_Rainfall*.tif"), recursive=True)
    
    if not terrain_files or not ndvi_files or not rainfall_files:
        print("Missing some .tif files in the data directory!")
        print(f"Terrain files found: {len(terrain_files)}")
        print(f"NDVI files found: {len(ndvi_files)}")
        print(f"Rainfall files found: {len(rainfall_files)}")
        return None

    print(f"Generating {num_samples} data points using real Earth Observation data...")
    points = []
    features = {
        "elevation": [],
        "slope": [],
        "aspect": [],
        "ndvi": [],
        "rainfall_mm": [],
        "soil_type": [],
        "landslide_occurred": []
    }
    
    generated = 0
    attempts = 0
    max_attempts = num_samples * 10
    
    while generated < num_samples and attempts < max_attempts:
        attempts += 1
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        point = Point(x, y)
        
        if leh_geom.contains(point):
            # Extract Raster Values
            terrain_val = get_pixel_value(x, y, terrain_files)
            if terrain_val is None:
                continue # Point might be in an empty spot or masked
                
            elevation = float(terrain_val[0])
            slope = float(terrain_val[1])
            aspect = float(terrain_val[2])
            
            # Filter out invalid values (e.g. nodata)
            if elevation < -500 or slope < 0 or aspect < 0:
                continue
                
            ndvi_val = get_pixel_value(x, y, ndvi_files)
            if ndvi_val is None:
                continue
            ndvi = float(ndvi_val[0])
            
            rain_val = get_pixel_value(x, y, rainfall_files)
            if rain_val is None:
                continue
            rainfall = float(rain_val[0])
            
            # Since we don't have soil data, we keep it random categorical for now
            soil_type = np.random.randint(1, 5)
            
            # Simulate historical ground truth for training purposes based on the *real* extracted features
            # A simplistic heuristic: steep slopes + heavy rain + lack of vegetation = landslide
            risk_score = (slope / 90.0) * 0.4 + (min(rainfall, 100) / 100.0) * 0.4 + ((1.0 - ndvi) / 2.0) * 0.2
            # Add some random noise
            risk_score += np.random.normal(0, 0.1)
            occurred = 1 if risk_score > 0.55 else 0
            
            points.append(point)
            features["elevation"].append(elevation)
            features["slope"].append(slope)
            features["aspect"].append(aspect)
            features["ndvi"].append(ndvi)
            features["rainfall_mm"].append(rainfall)
            features["soil_type"].append(soil_type)
            features["landslide_occurred"].append(occurred)
            
            generated += 1
            if generated % 100 == 0:
                print(f"Processed {generated}/{num_samples} points...")
                
    if generated < num_samples:
        print(f"Warning: Only able to generate {generated} valid points out of {num_samples}.")

    # Create DataFrame
    df = pd.DataFrame(features)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=points, crs=aoi_gdf.crs)
    
    output_path = os.path.join(config.BASE_DIR, "training_data.csv")
    gdf.drop(columns='geometry').to_csv(output_path, index=False)
    print(f"Data generated and saved to {output_path}")
    return gdf

if __name__ == "__main__":
    generate_real_data_for_leh(2000)
