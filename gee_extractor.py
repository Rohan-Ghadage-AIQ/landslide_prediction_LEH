import os
import ee
import geopandas as gpd
import config

# Replace this with your Google Cloud Project ID if you haven't set a default one via CLI
GEE_PROJECT_ID = 'job-portal-464909' 

def authenticate_and_initialize():
    """Authenticates and initializes Google Earth Engine."""
    try:
        if GEE_PROJECT_ID:
            ee.Initialize(project=GEE_PROJECT_ID)
        else:
            ee.Initialize()
        print("Earth Engine initialized successfully.")
    except ee.ee_exception.EEException as e:
        if 'no project found' in str(e).lower():
            print("\n" + "="*60)
            print("ERROR: Google Earth Engine requires a Cloud Project ID.")
            print("Please do ONE of the following:")
            print("1. Open 'gee_extractor.py' and set GEE_PROJECT_ID = 'your-project-id'")
            print("2. OR run this in your terminal: earthengine set_project <your-project-id>")
            print("="*60 + "\n")
        raise e
    except Exception as e:
        print("Earth Engine not authorized. Please run 'earthengine authenticate' in your terminal.")
        raise e

def export_to_drive(image, description, folder, region, scale=30):
    """Exports an Earth Engine image to Google Drive."""
    print(f"Starting export for {description}...")
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        fileFormat='GeoTIFF',
        maxPixels=1e13
    )
    task.start()
    print(f"Task {description} started. Check your Google Earth Engine Task Manager or Google Drive.")

def extract_data_for_leh():
    """Extracts DEM, NDVI, and Rainfall data for the Leh region."""
    authenticate_and_initialize()
    
    # 1. Load the shapefile to get the bounding box/geometry
    print(f"Loading shapefile from {config.AOI_SHP_PATH}...")
    aoi_gdf = gpd.read_file(config.AOI_SHP_PATH)
    
    # Convert geopandas geometry to Earth Engine geometry
    # Get the bounding box of the shapefile
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    ee_geometry = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    
    folder_name = "Landslide_Data_Leh"
    
    # 2. Extract DEM (Elevation), Slope, and Aspect
    print("Fetching SRTM Elevation data...")
    srtm = ee.Image("USGS/SRTMGL1_003").clip(ee_geometry)
    elevation = srtm.select('elevation')
    slope = ee.Terrain.slope(elevation)
    aspect = ee.Terrain.aspect(elevation)
    
    # Combine terrain features into one multi-band image and cast to float to prevent type inconsistency
    terrain_image = ee.Image.cat([elevation, slope, aspect]).toFloat()
    export_to_drive(terrain_image, "Leh_Terrain_DEM_Slope_Aspect", folder_name, ee_geometry.getInfo()['coordinates'], scale=30)
    
    # 3. Extract Sentinel-2 NDVI (Median composite over a year)
    print("Fetching Sentinel-2 NDVI data...")
    # Using 2023 data as an example baseline
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(ee_geometry) \
        .filterDate('2023-01-01', '2023-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median() \
        .clip(ee_geometry)
        
    # Calculate NDVI: (NIR - Red) / (NIR + Red) -> (B8 - B4) / (B8 + B4)
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    export_to_drive(ndvi, "Leh_Sentinel2_NDVI", folder_name, ee_geometry.getInfo()['coordinates'], scale=10)
    
    # 4. Extract Rainfall (CHIRPS Daily - e.g., accumulated rainfall for a typical monsoon month)
    print("Fetching CHIRPS Precipitation data...")
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterBounds(ee_geometry) \
        .filterDate('2023-07-01', '2023-07-31') \
        .sum() \
        .clip(ee_geometry) \
        .rename('rainfall_mm')
        
    # CHIRPS is lower resolution (approx 5km), so we use scale=5000
    export_to_drive(chirps, "Leh_CHIRPS_Rainfall", folder_name, ee_geometry.getInfo()['coordinates'], scale=5000)

if __name__ == "__main__":
    extract_data_for_leh()
    print("All tasks submitted to Earth Engine. You can monitor them at: https://code.earthengine.google.com/tasks")
