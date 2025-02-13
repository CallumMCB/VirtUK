import pandas as pd
from VirtUK import paths

def process_communal_residents(fp, hierarchy_fp, output_fp):
    # Read the communal residents data
    df = pd.read_csv(fp)

    # Read the hierarchy data (area to MSOA mapping)
    hierarchy_df = pd.read_csv(hierarchy_fp)

    # Merge the hierarchy data to add MSOA to the communal residents data
    df_merged = df.merge(hierarchy_df[['area', 'msoa']], on='area', how='left')

    # Save the data with MSOA added, ensuring each area is on its own row
    df_merged.to_csv(output_fp, index=False)

    print(f"Processed data saved to: {output_fp}")

# Define file paths
hierarchy_fp = f"{paths.data_path}/input/geography/oa_msoa_lad_regions.csv"
communal_fp = f"{paths.data_path}/input/households/communal_establishments/communal_residents_oa.csv"
output_fp = f"{paths.data_path}/input/households/communal_establishments/communal_residents_msoa_oa.csv"

# Process the file
process_communal_residents(communal_fp, hierarchy_fp, output_fp)
