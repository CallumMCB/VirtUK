import pandas as pd
from tqdm import tqdm
from VirtUK import paths

tqdm.pandas()  # Enable tqdm for pandas

# Filepaths
file_path = f'{paths.data_path}/raw_data/households/student_accommodation.csv'
oa_output_fp = f'{paths.data_path}/input/households/communal_establishments/oa_student_accommodation.csv'
msoa_output_fp = f'{paths.data_path}/input/households/communal_establishments/msoa_student_accommodation.csv'
hierarchy_fp = f'{paths.data_path}/input/geography/oa_msoa_lad_regions.csv'

# Read the hierarchy and data files
hierarchy_df = pd.read_csv(hierarchy_fp)
data = pd.read_csv(file_path)

# --------------------------
# Process OA-level data
# --------------------------

# Step 1: Drop the 'Output Areas' column
print("Dropping 'Output Areas' column...")
data = data.drop(columns=['Output Areas'])

# Step 2: Rename 'Output Areas Code' to 'area'
print("Renaming 'Output Areas Code' to 'area'...")
data = data.rename(columns={'Output Areas Code': 'area'})

# Step 3: Set 'area' as the index column
print("Setting 'area' as the index column...")
data = data.set_index('area')

# Step 4: Rename 'Student accommodation type (7 categories)' to 'accommodation type'
data = data.rename(columns={'Student accommodation type (7 categories)': 'accommodation type'})

# Step 5: Clean the 'accommodation type' values by removing unwanted prefixes
print("Cleaning 'accommodation type' values...")
data['accommodation type'] = data['accommodation type'].str.replace(
    r'^(Living in a|Living in an|living with)\s+', '', case=False, regex=True
)

# Step 6: Pivot 'Age (7 categories)' to columns with the Observation data.
print("Pivoting 'Age (7 categories)' to columns...")
processed_data = data.pivot_table(
    index=['area', 'accommodation type'],
    columns='Age (7 categories)',
    values='Observation',
    aggfunc='sum'
).reset_index()

# Step 7: Rename the age columns to the desired 'xx-yy' format
age_mapping = {
    'Aged 4 years and under': '0-4',
    'Aged 5 to 15 years': '5-15',
    'Aged 16 to 17 years': '16-17',
    'Aged 18 to 20 years': '18-20',
    'Aged 21 to 24 years': '21-24',
    'Aged 25 to 29 years': '25-29',
    'Aged 30 years and over': '30+'
}
processed_data = processed_data.rename(columns=age_mapping)

# Step 8: Reorder the columns.
age_order = ['0-4', '5-15', '16-17', '18-20', '21-24', '25-29', '30+']
columns_order = ['area', 'accommodation type'] + age_order
processed_data = processed_data[columns_order]

# Step 9: Drop the 'Age (7 categories) Code' column if it exists (ignore error if it doesn't)
print("Dropping 'Age (7 categories) Code' column if it exists...")
processed_data = processed_data.drop(columns=['Age (7 categories) Code'], errors='ignore')

# Save the OA-level data
processed_data.to_csv(oa_output_fp, index=False)
print("OA-level data saved to:", oa_output_fp)

# --------------------------
# Create msoa-level data by aggregating over areas
# --------------------------

print("Merging OA data with hierarchy information to map each area to its msoa...")

# IMPORTANT:
# We assume that the hierarchy dataframe contains at least two columns:
#   - 'area': matching the output area code from processed_data
#   - 'msoa': the corresponding msoa code/name for that area
#
# Adjust the column names if your hierarchy file uses different names.
merged_data = processed_data.merge(
    hierarchy_df[['msoa', 'area']],   # adjust if needed
    on='area',
    how='left'
)

# Group by msoa and accommodation type, summing the age columns.
# If you wish to sum over accommodation types as well (i.e. drop accommodation type),
# you could group solely by 'msoa'.
msoa_data = merged_data.groupby(['msoa', 'accommodation type'], as_index=False)[age_order].sum()

# Optionally, if you want to pivot the result so that each msoa is a row and each
# accommodation type is a column (or vice versa), you could further pivot msoa_data.
# Here we leave it as one row per combination.
msoa_data.to_csv(msoa_output_fp, index=False)
print("MSOA-level data saved to:", msoa_output_fp)