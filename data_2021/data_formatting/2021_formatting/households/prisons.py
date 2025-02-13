import pandas as pd
from tabulate import tabulate
from VirtUK import paths

# Directory paths
households_dir = f"{paths.data_path}/input/households"
communal_establishments_dir = f"{households_dir}/communal_establishments"

# Files
prisons_file = f"{paths.data_path}/raw_data/households/prisons.xlsx"
postcode_oa_file = f"{paths.data_path}/raw_data/geography/PCD_OA21_LSOA21_MSOA21_LAD_AUG24_UK_LU.csv"
resident_type_msoa_file = f"{communal_establishments_dir}/resident_type_msoa.csv"
communal_residents_msoa_oa_file = f"{communal_establishments_dir}/communal_residents_msoa_oa.csv"
output_file = f"{communal_establishments_dir}/prisons/prisons_formatted.csv"

# Step 1: Load the prisons data, skipping the first row and limiting to row 133
prisons_df = pd.read_excel(prisons_file, sheet_name='Prison Estate', skiprows=1, nrows=132)

# Clean column names to remove any leading/trailing whitespace
prisons_df.columns = prisons_df.columns.str.strip()

# Step 2: Keep only the columns 'Prison', 'Predominant Function', 'Designation', 'Postal Address'
prisons_df = prisons_df[['Prison', 'Predominant Function', 'Designation', 'Postal Address']]

# Step 3: Filter out rows where the name of the prison is 'Prison'
prisons_df = prisons_df[prisons_df['Prison'].str.strip().str.lower() != 'prison']

# Step 4: Extract the postcode from the postal address
def extract_postcode(addr):
    if pd.isna(addr):
        return None
    lines = addr.split('\n')
    # Extract postcode assuming last line and comma separation:
    postcode = lines[-1].split(',')[-1].upper().replace(" ", "")
    return postcode

prisons_df['Postcode'] = prisons_df['Postal Address'].apply(extract_postcode)

# Step 5: Load the postcode to OA/MSOA mapping file
# The file contains columns: 'pcd7' (postcode), 'oa21cd' (output area), 'msoa21cd' (MSOA)
postcode_oa_df = pd.read_csv(postcode_oa_file, encoding='ISO-8859-1')

# Standardize postcode column for matching
postcode_oa_df['pcd7'] = postcode_oa_df['pcd7'].str.upper().str.replace(" ", "", regex=False)

# Merge prisons data with the postcode to OA/MSOA mapping
prisons_df = prisons_df.merge(
    postcode_oa_df[['pcd7', 'oa21cd', 'msoa21cd']],
    left_on='Postcode',
    right_on='pcd7',
    how='left'
).drop(columns='pcd7')

# Dictionary to manually add missing coordinates and areas
manual_coords = {
    'L93DF0': {'oa21cd': 'E00182973', 'msoa21cd': 'E02001349'},
    'PL206R0': {'oa21cd': 'E00180228', 'msoa21cd': 'E02004231'},
    'DN228EU0': {'oa21cd': 'E00142851', 'msoa21cd': 'E02005837'},
    'B976QS': {'oa21cd': 'E00181370', 'msoa21cd': 'E02006709'},
    'DT51DL': {'oa21cd': 'E00171471', 'msoa21cd': 'E02004288'}
}

def add_manual_data(row):
    if row['Postcode'] in manual_coords:
        row['oa21cd'] = manual_coords[row['Postcode']]['oa21cd']
        row['msoa21cd'] = manual_coords[row['Postcode']]['msoa21cd']
    return row

prisons_df = prisons_df.apply(add_manual_data, axis=1)

# Step 6: Load the communal_residents_msoa_oa.csv file and merge 'total communal residents'
# Assuming communal_residents_msoa_oa.csv has columns: ['area', 'communal residents']
communal_residents_df = pd.read_csv(communal_residents_msoa_oa_file)

# Merge based on OA21CD = 'area'
prisons_df = prisons_df.merge(
    communal_residents_df[['area', 'communal residents']],
    left_on='oa21cd',
    right_on='area',
    how='left'
).drop(columns='area')

prisons_df.rename(columns={'communal residents': 'total communal residents'}, inplace=True)

# Step 7: Load resident_type_msoa.csv and merge 'Other: Prison service' based on MSOA
# Assuming resident_type_msoa.csv has columns: ['msoa', 'Other: Prison service']
resident_type_msoa_df = pd.read_csv(resident_type_msoa_file)

prisons_df = prisons_df.merge(
    resident_type_msoa_df[['msoa', 'Other: Prison service']],
    left_on='msoa21cd',
    right_on='msoa',
    how='left'
).drop(columns='msoa')

# Rename the column for clarity (optional)
prisons_df.rename(columns={'Other: Prison service': 'other_prison_service'}, inplace=True)

# Select final columns
output_columns = [
    'Prison',
    'Predominant Function',
    'Designation',
    'Postcode',
    'oa21cd',
    'msoa21cd',
    'total communal residents',
    'other_prison_service'
]

final_df = prisons_df[output_columns].copy()
final_df.rename(columns={'oa21cd': 'OA21CD', 'msoa21cd': 'MSOA21CD'}, inplace=True)

# Print the formatted table for verification
print(tabulate(final_df, headers='keys', tablefmt='fancy_grid'))

# Step 8: Output to CSV
final_df.to_csv(output_file, index=False)
print(f"Formatted prison data saved to {output_file}")
