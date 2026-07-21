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
            
            # Geotechnical & Hydrological Slope Stability Model (AHP Formulation)
            # 1. Non-linear Slope Factor (Slopes > 25° exponentially increase landslide susceptibility)
            slope_factor = np.sin(np.radians(slope)) ** 2.0
            
            # 2. Rainfall Saturation Trigger (Non-linear saturation response)
            rain_factor = min(rainfall / 150.0, 1.0)
            
            # 3. Vegetation & Root Cohesion Factor (Bare rock/scree with NDVI < 0.2 has high vulnerability)
            veg_factor = max(0.0, (0.4 - ndvi) / 0.4) if ndvi < 0.4 else 0.0
            
            # 4. High-Altitude Himalayan Freeze-Thaw Factor (Leh altitudes > 3500m face permafrost instability)
            alt_factor = 1.15 if elevation > 3800 else 1.0
            
            # Combined Geotechnical Risk Score
            risk_score = (0.45 * slope_factor + 0.35 * rain_factor + 0.20 * veg_factor) * alt_factor
            
            # Add subtle natural environmental variation
            risk_score += np.random.normal(0, 0.03)
            occurred = 1 if risk_score > 0.38 else 0
            
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

def generate_training_data_from_gsi_points(gsi_file_path: str):
    """
    Processes a real GSI Landslide Inventory file (GeoJSON, Shapefile, or CSV with lat/lon)
    and extracts satellite raster features at exact landslide coordinates.
    Generates balanced positive (1) and negative (0) training samples.
    """
    print(f"Loading GSI Landslide Inventory from {gsi_file_path}...")
    if not os.path.exists(gsi_file_path):
        raise FileNotFoundError(f"GSI file not found: {gsi_file_path}")
        
    # Read file based on format
    if gsi_file_path.endswith('.csv'):
        df_gsi = pd.read_csv(gsi_file_path)
        # Expecting latitude/longitude columns
        lat_col = [c for c in df_gsi.columns if 'lat' in c.lower()][0]
        lon_col = [c for c in df_gsi.columns if 'lon' in c.lower() or 'lng' in c.lower()][0]
        geometry = [Point(xy) for xy in zip(df_gsi[lon_col], df_gsi[lat_col])]
        gsi_gdf = gpd.GeoDataFrame(df_gsi, geometry=geometry, crs="EPSG:4326")
    else:
        gsi_gdf = gpd.read_file(gsi_file_path)
        
    # Locate Data Rasters
    data_dir = os.path.join(config.BASE_DIR, "data")
    terrain_files = glob.glob(os.path.join(data_dir, "**", "Leh_Terrain_DEM_Slope_Aspect*.tif"), recursive=True)
    ndvi_files = glob.glob(os.path.join(data_dir, "**", "Leh_Sentinel2_NDVI*.tif"), recursive=True)
    rainfall_files = glob.glob(os.path.join(data_dir, "**", "Leh_CHIRPS_Rainfall*.tif"), recursive=True)
    
    features = {
        "elevation": [], "slope": [], "aspect": [], "ndvi": [], "rainfall_mm": [], "soil_type": [], "landslide_occurred": []
    }
    
    # Process positive GSI points (Landslides = 1)
    pos_count = 0
    for idx, row in gsi_gdf.iterrows():
        pt = row.geometry
        lon, lat = pt.x, pt.y
        
        terrain_val = get_pixel_value(lon, lat, terrain_files)
        ndvi_val = get_pixel_value(lon, lat, ndvi_files)
        rain_val = get_pixel_value(lon, lat, rainfall_files)
        
        if terrain_val is not None and ndvi_val is not None and rain_val is not None:
            features["elevation"].append(float(terrain_val[0]))
            features["slope"].append(float(terrain_val[1]))
            features["aspect"].append(float(terrain_val[2]))
            features["ndvi"].append(float(ndvi_val[0]))
            features["rainfall_mm"].append(float(rain_val[0]))
            features["soil_type"].append(np.random.randint(1, 5))
            features["landslide_occurred"].append(1)
            pos_count += 1
            
    print(f"Extracted features for {pos_count} real GSI landslide points.")
    
    # Process matching negative points (Stable terrain / Landslides = 0)
    print(f"Sampling {pos_count * 2} non-landslide (stable terrain) negative points...")
    aoi_gdf = gpd.read_file(config.AOI_SHP_PATH)
    leh_geom = aoi_gdf.geometry.iloc[0]
    minx, miny, maxx, maxy = leh_geom.bounds
    
    neg_count = 0
    attempts = 0
    target_neg = pos_count * 2
    while neg_count < target_neg and attempts < target_neg * 20:
        attempts += 1
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        point = Point(x, y)
        
        if leh_geom.contains(point):
            terrain_val = get_pixel_value(x, y, terrain_files)
            ndvi_val = get_pixel_value(x, y, ndvi_files)
            rain_val = get_pixel_value(x, y, rainfall_files)
            
            if terrain_val is not None and ndvi_val is not None and rain_val is not None:
                slope = float(terrain_val[1])
                # Ensure negative samples are from relatively stable / low slope zones (< 20 degrees)
                if slope < 25.0:
                    features["elevation"].append(float(terrain_val[0]))
                    features["slope"].append(slope)
                    features["aspect"].append(float(terrain_val[2]))
                    features["ndvi"].append(float(ndvi_val[0]))
                    features["rainfall_mm"].append(float(rain_val[0]))
                    features["soil_type"].append(np.random.randint(1, 5))
                    features["landslide_occurred"].append(0)
                    neg_count += 1

    print(f"Sampled {neg_count} negative (stable terrain) points.")

    # Save output
    output_df = pd.DataFrame(features)
    output_path = os.path.join(config.BASE_DIR, "training_data.csv")
    output_df.to_csv(output_path, index=False)
    print(f"GSI balanced dataset created and saved to {output_path} (Total samples: {len(output_df)})")
    return output_df

if __name__ == "__main__":
    generate_real_data_for_leh(2000)

