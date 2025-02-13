import pandas as pd
from VirtUK import paths

# File paths
csv_fp = f'{paths.data_path}/raw_data/geography/PCD_OA21_LSOA21_MSOA21_LAD_AUG24_UK_LU.csv'
area_msoa_region_output_fp = f'{paths.data_path}/input/geography/oa_msoa_lad_regions.csv'
lad_lookup_output_fp = f'{paths.data_path}/input/geography/lad_lookup.csv'

# Load CSV in chunks to prevent memory overload
chunk_size = 100000  # Customize this depending on your memory capacity
chunks = []

print("Loading data in chunks...")

for chunk in pd.read_csv(csv_fp, encoding='ISO-8859-1', low_memory=False, chunksize=chunk_size):
    chunk['oa21cd'] = chunk['oa21cd'].astype(str)
    # Remove rows where 'oa21cd' starts with 'S0' or is 'nan'
    chunk_filtered = chunk[(chunk['oa21cd'] != 'nan') & (~chunk['oa21cd'].str.startswith('S0'))]
    chunks.append(chunk_filtered)

print("Concatenating chunks...")
df_filtered = pd.concat(chunks, ignore_index=True)

# Select specific columns and rename them
df_area_msoa = df_filtered.loc[:, ['oa21cd', 'msoa21cd', 'ladcd', 'ladnm']].rename(
    columns={'oa21cd': 'area', 'msoa21cd': 'msoa', 'ladnm': 'lad', 'ladcd': 'lad_code'}
)
df_area_msoa['lad'] = df_area_msoa['lad'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else x)
df_area_msoa.sort_values(by='area', inplace=True)
df_area_msoa = df_area_msoa.drop_duplicates(subset='area')

# Manually add missing lads for specific lad_codes
missing_lads = {
    'E06000063': 'Cumberland',
    'E06000064': 'Westmorland and Furness',
    'E06000065': 'North Yorkshire',
    'E06000066': 'Somerset'
}

df_area_msoa['lad'] = df_area_msoa.apply(
    lambda row: missing_lads[row['lad_code']] if pd.isna(row['lad']) and row['lad_code'] in missing_lads else row['lad'],
    axis=1
)

df_area_msoa['lad'] = df_area_msoa['lad'].replace('Bournemouth', 'Bournemouth, Christchurch and Poole')

# Print any still missing lads
missing_lads_codes = df_area_msoa[df_area_msoa['lad'].isna()]['lad_code'].unique()
if len(missing_lads_codes) > 0:
    print(f"The following lad_codes are still missing names: {missing_lads_codes}")
else:
    print("No missing lads.")

# Adding missing output area rows
missing_output_areas = [
    {'area': 'E00008771', 'msoa': 'E02000364'},
    {'area': 'E00017435', 'msoa': 'E02000704'},
    {'area': 'E00171320', 'msoa': 'E02006878'},
    {'area': 'E00175048', 'msoa': 'E02000731'},
    {'area': 'E00176268', 'msoa': 'E02000366'},
    {'area': 'E00177379', 'msoa': 'E02002145'},
    {'area': 'E00177667', 'msoa': 'E02000649'},
    {'area': 'E00182141', 'msoa': 'E02006928'},
    {'area': 'E00182250', 'msoa': 'E02006928'},
    {'area': 'E00182311', 'msoa': 'E02006992'},
    {'area': 'E00182331', 'msoa': 'E02006991'},
    {'area': 'E00184428', 'msoa': 'E02000453'},
    {'area': 'E00185486', 'msoa': 'E02002785'},
    {'area': 'E00185550', 'msoa': 'E02002792'},
    {'area': 'E00185779', 'msoa': 'E02000020'},
    {'area': 'E00187113', 'msoa': 'E02002392'},
    {'area': 'E00187556', 'msoa': 'E02006904'},
    {'area': 'E00187835', 'msoa': 'E02003255'},
    {'area': 'E00187909', 'msoa': 'E02007054'},
    {'area': 'E00189115', 'msoa': 'E02007072'},
    {'area': 'E00189826', 'msoa': 'E02005812'},
    {'area': 'E00190444', 'msoa': 'E02000187'},
    {'area': 'W00010414', 'msoa': 'W02000184'},
    {'area': 'E00184703', 'msoa': 'E02005523'}
]

missing_output_areas_df = pd.DataFrame(missing_output_areas)

# Function to assign lad_code and lad based on a matching msoa
def assign_lad_info(row, reference_df):
    matching_row = reference_df[reference_df['msoa'] == row['msoa']].iloc[0]
    row['lad_code'] = matching_row['lad_code']
    row['lad'] = matching_row['lad']
    return row

missing_output_areas_df = missing_output_areas_df.apply(assign_lad_info, reference_df=df_area_msoa, axis=1)

# Merge missing rows into the main dataframe
df_area_msoa = pd.concat([df_area_msoa, missing_output_areas_df], ignore_index=False)
df_area_msoa.sort_values(by='area', inplace=True)
df_area_msoa.reset_index(drop=True, inplace=True)

areas_region_fp = f'{paths.data_path}/raw_data/geography/OA21_RGN22_LU.csv'
areas_region_df = pd.read_csv(areas_region_fp, low_memory=False)
df_area_msoa_lad_region = pd.merge(df_area_msoa, areas_region_df, left_on='area', right_on='oa21cd', how='left')
df_area_msoa_lad_region.drop(columns=['oa21cd', 'rgn22nmw'], inplace=True)
df_area_msoa_lad_region = df_area_msoa_lad_region.rename(columns={'rgn22cd': 'region_code', 'rgn22nm': 'region'})

# *** Modification: Remove the LAD name from the main file ***
# Keep only the lad_code in the main file.
df_area_msoa_lad_region = df_area_msoa_lad_region.drop(columns=['lad'])

# Create a separate lookup file for lad_code <-> lad mapping.
df_lad_lookup = df_area_msoa[['lad_code', 'lad']].drop_duplicates()
df_lad_lookup.sort_values(by='lad_code', inplace=True)

print("Saving the modified CSVs...")
df_area_msoa_lad_region.to_csv(area_msoa_region_output_fp, index=False)
df_lad_lookup.to_csv(lad_lookup_output_fp, index=False)

print(f'Modified CSV file saved to: {area_msoa_region_output_fp}')
print(f'LAD lookup CSV file saved to: {lad_lookup_output_fp}')
