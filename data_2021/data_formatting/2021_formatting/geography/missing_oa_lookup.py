import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from VirtUK import paths

# File paths
areas_coordinates_fp = f'{paths.data_path}/input/geography/oa_coordinates.csv'
msoa_geojson_fp = f'{paths.data_path}/input/geography/msoa_boundaries.geojson'

# Function to find MSOA for given output area codes based on coordinates
def find_msoa_for_output_areas(output_area_codes, coordinates_fp, geojson_fp):
    # Load coordinates CSV
    coordinates_df = pd.read_csv(coordinates_fp)
    # Filter coordinates for specified output area codes
    filtered_coords = coordinates_df[coordinates_df['area'].isin(output_area_codes)]

    # Load MSOA geojson
    msoa_gdf = gpd.read_file(geojson_fp)

    # Prepare dictionary to store results
    results = []

    # Iterate over filtered coordinates and find corresponding MSOA
    for _, row in filtered_coords.iterrows():
        point = Point(row['longitude'], row['latitude'])
        for _, msoa_row in msoa_gdf.iterrows():
            if msoa_row['geometry'].contains(point):
                results.append({'area': row['area'], 'msoa': msoa_row['MSOA21CD']})
                break

    return results

# Example usage of the function
output_area_codes = ['E00184703', ]  # Replace with desired output area codes
results = find_msoa_for_output_areas(output_area_codes, areas_coordinates_fp, msoa_geojson_fp)
print(results)
