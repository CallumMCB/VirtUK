import pandas as pd
import numpy as np
from tqdm import tqdm
from VirtUK import paths
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint

# Define the file paths
filtered_file = f'{paths.data_path}/raw_data/households/care_home_occupancy.csv'
lad_region_file = f'{paths.data_path}/input/geography/oa_msoa_lad_regions.csv'
lad_boundaries_file = f'{paths.data_path}/input/geography/lad_boundaries.geojson'
msoa_boundaries_file = f'{paths.data_path}/input/geography/msoa_boundaries.geojson'
oa_boundaries_file = f'{paths.data_path}/input/geography/oa_boundaries.geojson'
output_file = f'{paths.data_path}/input/households/communal_establishments/care_homes/occupancy.csv'

# Load filtered data
with tqdm(total=1, desc="Reading filtered CSV file", dynamic_ncols=True) as pbar:
    df_filtered = pd.read_csv(filtered_file)
    pbar.update(1)

chunk_list = []
for chunk in pd.read_csv(lad_region_file, chunksize=100000):
    chunk_list.append(chunk)
lad_region_df = pd.concat(chunk_list, axis=0)
lad_boundaries = gpd.read_file(lad_boundaries_file)
msoa_boundaries = gpd.read_file(msoa_boundaries_file)
oa_boundaries = gpd.read_file(oa_boundaries_file)

# Drop rows where coordinates are missing
df_filtered = df_filtered.dropna(subset=['Location Latitude', 'Location Longitude']).reset_index(drop=True)

# Prepare dictionaries for faster lookup
lad_region_df = lad_region_df.reset_index(drop=True)
oa_to_msoa = lad_region_df.set_index('area')['msoa'].to_dict()

# Ensure valid region names are used
regions = lad_region_df['region'].dropna().unique()
all_results = []


# Function to find the nearest geometry if no match is found
def find_nearest_geometry(boundaries, latitude, longitude):
    point = Point(longitude, latitude)
    all_points = MultiPoint([boundary['geometry'].centroid for _, boundary in boundaries.iterrows()])
    nearest_geom = nearest_points(point, all_points)[1]
    for _, boundary in boundaries.iterrows():
        if boundary['geometry'].centroid.equals(nearest_geom):
            return boundary
    return None


# Vectorized function to process LADs
def process_lads(df_filtered, lad_boundaries):
    # Create a GeoDataFrame from filtered DataFrame
    gdf_points = gpd.GeoDataFrame(
        df_filtered,
        geometry=gpd.points_from_xy(df_filtered['Location Longitude'],
                                    df_filtered['Location Latitude']),
        crs="EPSG:4326"
    )
    # Spatial join to find LAD boundaries
    gdf_joined = gpd.sjoin(gdf_points, lad_boundaries, how="left", predicate='within')

    # Extract relevant fields from the join result
    df_filtered['Location Local Authority'] = gdf_joined['LAD24NM']
    df_filtered['lad'] = gdf_joined['LAD24CD']

    return df_filtered


# Vectorized function to process MSOAs and OAs
def process_msoas_and_oas(df_filtered, msoa_boundaries, oa_boundaries, lad_region_df):
    all_results = []

    # Group by LAD and process each LAD sequentially
    for lad_code, df_group in df_filtered.groupby('lad'):
        if df_group.empty:
            print(f"Skipping LAD Code {lad_code} because the group is empty.")
            continue

        possible_msoas = lad_region_df[lad_region_df['lad_code'] == lad_code]['msoa'].unique()
        msoa_boundaries_filtered = msoa_boundaries[msoa_boundaries['MSOA21CD'].isin(possible_msoas)]

        # Create a GeoDataFrame from the group
        gdf_points = gpd.GeoDataFrame(
            df_group,
            geometry=gpd.points_from_xy(df_group['Location Longitude'],
                                        df_group['Location Latitude']),
            crs="EPSG:4326"
        )

        # Spatial join to find MSOA boundaries
        gdf_joined = gpd.sjoin(gdf_points, msoa_boundaries_filtered, how="left", predicate='within')

        # Extract relevant fields from the join result
        df_group['msoa'] = gdf_joined['MSOA21CD']

        # Process OAs for the MSOAs found
        possible_oas = lad_region_df[lad_region_df['msoa'].isin(df_group['msoa'].unique())]['area'].unique()
        oa_boundaries_filtered = oa_boundaries[oa_boundaries['OA21CD'].isin(possible_oas)]

        # Spatial join to find OA boundaries
        gdf_joined_oa = gpd.sjoin(gdf_points, oa_boundaries_filtered, how="left", predicate='within')

        # Extract relevant fields from the join result
        df_group['area'] = gdf_joined_oa['OA21CD']
        all_results.append(df_group)

    if len(all_results) > 0:
        return pd.concat(all_results, ignore_index=True)
    else:
        print("No valid groups to concatenate. Returning an empty DataFrame.")
        return pd.DataFrame(columns=df_filtered.columns)


# Group by region and process each region sequentially
with tqdm(total=len(regions), desc="Processing regions", position=0, leave=True, dynamic_ncols=True) as pbar:
    for region in regions:
        df_region = df_filtered[df_filtered['Location Region'] == region].copy()

        # Process LADs for the region
        df_region = process_lads(df_region, lad_boundaries)

        # Process MSOAs and OAs within each LAD
        df_region = process_msoas_and_oas(df_region, msoa_boundaries, oa_boundaries, lad_region_df)

        all_results.append(df_region)
        pbar.update(1)

# Concatenate all results
if len(all_results) > 0:
    df_filtered = pd.concat(all_results, ignore_index=True)
else:
    print("No valid regions to concatenate. Returning an empty DataFrame.")
    df_filtered = pd.DataFrame(columns=df_filtered.columns)

# --- Additional Logic: Compute Mean and Variance of % Occupancy ---

# Identify all columns that contain '%occupancy'
occupancy_columns = [col for col in df_filtered.columns if 'extracted_value' in col]
percentage_occupancy_columns = [col for col in df_filtered.columns if '%occupancy' in col]
probability_columns = [col for col in df_filtered.columns if 'probability' in col]

if occupancy_columns:
    # Compute the average occupancy (multiplied by 100) and round the result
    df_filtered['occupancy_mean'] = (df_filtered[occupancy_columns].mean(axis=1)).round()
    df_filtered['%occupancy_mean'] = (df_filtered[percentage_occupancy_columns].mean(axis=1) * 100).round()
    df_filtered['probability_mean'] = (df_filtered[probability_columns].mean(axis=1)).round(5)
    # Compute the std of occupancy values
    df_filtered['occupancy_std'] = df_filtered[occupancy_columns].std(axis=1).round()
    df_filtered['%occupancy_std'] = ((df_filtered[percentage_occupancy_columns].std(axis=1))*100).round(4)
    df_filtered['probability_std'] = df_filtered[probability_columns].std(axis=1).round(5)
else:
    print("No occupancy columns found containing '%occupancy'.")

print(df_filtered['probability_mean'].mean(), df_filtered['probability_std'].mean())

df_filtered = df_filtered.drop(columns=occupancy_columns + percentage_occupancy_columns + probability_columns)

# Save the filtered DataFrame to a CSV file
df_filtered.to_csv(output_file, index=False)
print(f"Filtered data saved to: {output_file}")
