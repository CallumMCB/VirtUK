# split_data_by_region.py

import os
import geopandas as gpd
import pandas as pd

from VirtUK import paths

DATA_PATH = paths.data_path

# Define base directories for each category.
geo_base_dir = os.path.join(DATA_PATH, 'input', 'geography')
geo_raw_dir = os.path.join(DATA_PATH, 'raw_data', 'geography')
demo_base_dir = os.path.join(DATA_PATH, 'input', 'demography')
communal_base_dir = os.path.join(DATA_PATH, 'input', 'households', 'communal_establishments')
nhs_base_dir = os.path.join(DATA_PATH, 'input', 'NHS_trusts')

# Define file paths for the original files.
# Geography files:
oa_boundaries_path = os.path.join(geo_raw_dir, 'oa_boundaries.geojson')
msoa_boundaries_path = os.path.join(geo_raw_dir, 'msoa_boundaries.geojson')
lad_boundaries_path = os.path.join(geo_raw_dir, 'lad_boundaries.geojson')
regions_mapping_path = os.path.join(geo_base_dir, 'oa_msoa_lad_regions.csv')
lad_names = os.path.join(geo_base_dir, 'lad_lookup.csv')

# Demography files:
age_dist_dir = os.path.join(demo_base_dir, 'age_dist_2021')
total_age_msoa = os.path.join(age_dist_dir, 'msoa_total_population.csv')
female_age_msoa = os.path.join(age_dist_dir, 'msoa_female_population.csv')
five_year_ages_path = os.path.join(age_dist_dir, "5_year_ages_oa_(actual).csv")

# Communal establishments files:
care_homes_path = os.path.join(communal_base_dir, 'care_homes', 'occupancy_filled.csv')
resident_type_msoa_path = os.path.join(communal_base_dir, 'resident_type_msoa.csv')
female_residents_path = os.path.join(communal_base_dir, 'female_residents_msoa.csv')
male_residents_path = os.path.join(communal_base_dir, 'male_residents_msoa.csv')
staff_or_temporary_path = os.path.join(communal_base_dir, 'staff_or_temporary_msoa.csv')
# Additional communal establishments files:
communal_residents_msoa_oa_path = os.path.join(communal_base_dir, 'communal_residents_msoa_oa.csv')
establishment_type_msoa_path = os.path.join(communal_base_dir, 'establishment_type_msoa.csv')
oa_student_accommodation_path = os.path.join(communal_base_dir, 'oa_student_accommodation.csv')
msoa_student_accommodation_path = os.path.join(communal_base_dir, 'msoa_student_accommodation.csv')
prisons_formatted_path = os.path.join(communal_base_dir, 'prisons', 'prisons_formatted.csv')

# Prisons actual files (globally created):
age_group_table_path = os.path.join(communal_base_dir, 'prisons', 'age_group_table.csv')
custody_type_table_path = os.path.join(communal_base_dir, 'prisons', 'custody_type_table.csv')
nationality_group_table_path = os.path.join(communal_base_dir, 'prisons', 'nationality_group_table.csv')
offence_group_table_path = os.path.join(communal_base_dir, 'prisons', 'offence_group_table.csv')
ethnicity_group_table_path = os.path.join(communal_base_dir, 'prisons', 'ethnicity_group_table.csv')

# NHS files:
trusts_path = os.path.join(nhs_base_dir, 'unique_trust_locations.csv')
hospitals_path = os.path.join(nhs_base_dir, 'unique_hospital_locations.csv')
hospice_path = os.path.join(nhs_base_dir, 'unique_hospice_locations.csv')

# Load the original files.
print("Loading geography files...")
oa_boundaries = gpd.read_file(oa_boundaries_path)
msoa_boundaries = gpd.read_file(msoa_boundaries_path)
lad_boundaries = gpd.read_file(lad_boundaries_path)
lad_names = pd.read_csv(lad_names)
lad_names.set_index('lad_code', inplace=True)
df_regions = pd.read_csv(regions_mapping_path)

print("Loading demography files...")
df_total_age = pd.read_csv(total_age_msoa)
df_female_age = pd.read_csv(female_age_msoa)
df_five_year_ages = pd.read_csv(five_year_ages_path)

print("Loading communal establishments files...")
df_care_homes = pd.read_csv(care_homes_path)
df_resident_type_msoa = pd.read_csv(resident_type_msoa_path)
df_female_residents = pd.read_csv(female_residents_path)
df_male_residents = pd.read_csv(male_residents_path)
df_staff_or_temporary = pd.read_csv(staff_or_temporary_path)
# Load additional communal establishments files:
df_communal_residents_msoa_oa = pd.read_csv(communal_residents_msoa_oa_path)
df_establishment_type_msoa = pd.read_csv(establishment_type_msoa_path)
df_oa_student_accommodation = pd.read_csv(oa_student_accommodation_path)
df_msoa_student_accommodation = pd.read_csv(msoa_student_accommodation_path)
df_prisons_formatted = pd.read_csv(prisons_formatted_path)
df_prisons_formatted.rename(columns={'OA21CD': 'area', 'MSOA21CD': 'msoa'}, inplace=True)
df_prisons_formatted.to_csv(prisons_formatted_path, index=False)

print("Loading processed prisons actual files...")
df_age_group_global = pd.read_csv(age_group_table_path)
df_custody_type_global = pd.read_csv(custody_type_table_path)
df_nationality_group_global = pd.read_csv(nationality_group_table_path)
df_offence_group_global = pd.read_csv(offence_group_table_path)
df_ethnicity_group_global = pd.read_csv(ethnicity_group_table_path)

print("Loading NHS files...")
df_trusts = pd.read_csv(trusts_path)
df_hospitals = pd.read_csv(hospitals_path)
df_hospice = pd.read_csv(hospice_path)

# Get the unique regions from the mapping file.
unique_regions = df_regions['region'].unique()
print(f"Found regions: {unique_regions}")

# For each region, filter and save files into category-specific subdirectories.
for region in unique_regions:
    print(f"\nProcessing region: {region}")

    # Filter the mapping for this region.
    region_map = df_regions[df_regions['region'] == region]
    region_msoas = region_map['msoa'].unique()  # used for msoa-based filtering
    region_oas = region_map['area'].unique()  # used for oa-based filtering

    # --- Geography Files ---
    # Create a directory: [...]/input/geography/regions/<region>
    region_geo_dir = os.path.join(geo_base_dir, 'regions', region)
    os.makedirs(region_geo_dir, exist_ok=True)

    # Filter geography files using the mapping.
    region_oa_boundaries = oa_boundaries[oa_boundaries['OA21CD'].isin(region_map['area'])]
    region_msoa_boundaries = msoa_boundaries[msoa_boundaries['MSOA21CD'].isin(region_map['msoa'])]
    region_lad_boundaries = lad_boundaries[lad_boundaries['LAD24NM'].isin(lad_names.loc[region_map['lad_code'], 'lad'])]

    # Save the filtered geography files.
    region_oa_boundaries.to_file(os.path.join(region_geo_dir, 'oa_boundaries.geojson'), driver='GeoJSON')
    region_msoa_boundaries.to_file(os.path.join(region_geo_dir, 'msoa_boundaries.geojson'), driver='GeoJSON')
    region_lad_boundaries.to_file(os.path.join(region_geo_dir, 'lad_boundaries.geojson'), driver='GeoJSON')
    # Also save the region's mapping.
    region_map.to_csv(os.path.join(region_geo_dir, 'oa_msoa_lad_regions.csv'), index=False)

    # --- Demography Files ---
    # Create a directory: [...]/input/demography/regions/<region>
    region_demo_dir = os.path.join(demo_base_dir, 'regions', region)
    os.makedirs(region_demo_dir, exist_ok=True)

    # Filter demography files by msoa.
    region_total_age = df_total_age[df_total_age['msoa'].isin(region_msoas)]
    region_female_age = df_female_age[df_female_age['msoa'].isin(region_msoas)]
    # Save filtered demography files.
    region_total_age.to_csv(os.path.join(region_demo_dir, 'msoa_total_population.csv'), index=False)
    region_female_age.to_csv(os.path.join(region_demo_dir, 'msoa_female_population.csv'), index=False)

    # Filter the 5_year_ages_oa_(actual).csv file by output area.
    region_five_year_ages = df_five_year_ages[df_five_year_ages['area'].isin(region_oas)]
    region_five_year_ages.to_csv(os.path.join(region_demo_dir, '5_year_ages_oa_(actual).csv'), index=False)

    # --- Communal Establishments Files ---
    # Create a directory: [...]/input/households/communal_establishments/regions/<region>
    region_communal_dir = os.path.join(communal_base_dir, 'regions', region)
    os.makedirs(region_communal_dir, exist_ok=True)

    # Filter communal establishments files.
    region_care_homes = df_care_homes[df_care_homes['Location Region'] == region]
    region_resident_type_msoa = df_resident_type_msoa[df_resident_type_msoa['msoa'].isin(region_msoas)]
    region_female_residents = df_female_residents[df_female_residents['msoa'].isin(region_msoas)]
    region_male_residents = df_male_residents[df_male_residents['msoa'].isin(region_msoas)]
    region_staff_or_temporary = df_staff_or_temporary[df_staff_or_temporary['msoa'].isin(region_msoas)]
    # Additional communal establishments:
    region_communal_residents_msoa_oa = df_communal_residents_msoa_oa[df_communal_residents_msoa_oa['msoa'].isin(region_msoas)]
    region_establishment_type_msoa = df_establishment_type_msoa[df_establishment_type_msoa['msoa'].isin(region_msoas)]
    region_oa_student_accommodation = df_oa_student_accommodation[df_oa_student_accommodation['area'].isin(region_oas)]
    region_msoa_student_accommodation = df_msoa_student_accommodation[df_msoa_student_accommodation['msoa'].isin(region_msoas)]
    region_prisons_formatted = df_prisons_formatted[df_prisons_formatted['area'].isin(region_oas)]

    # Save communal establishments files.
    region_care_homes.to_csv(os.path.join(region_communal_dir, 'care_homes_occupancy_filled.csv'), index=False)
    region_resident_type_msoa.to_csv(os.path.join(region_communal_dir, 'resident_type_msoa.csv'), index=False)
    region_female_residents.to_csv(os.path.join(region_communal_dir, 'female_residents_msoa.csv'), index=False)
    region_male_residents.to_csv(os.path.join(region_communal_dir, 'male_residents_msoa.csv'), index=False)
    region_communal_residents_msoa_oa.to_csv(os.path.join(region_communal_dir, 'communal_residents_msoa_oa.csv'), index=False)
    region_staff_or_temporary.to_csv(os.path.join(region_communal_dir, 'staff_or_temporary_msoa.csv'), index=False)
    region_establishment_type_msoa.to_csv(os.path.join(region_communal_dir, 'establishment_type_msoa.csv'), index=False)
    region_oa_student_accommodation.to_csv(os.path.join(region_communal_dir, 'oa_student_accommodation.csv'), index=False)
    region_msoa_student_accommodation.to_csv(os.path.join(region_communal_dir, 'msoa_student_accommodation.csv'), index=False)
    region_prisons_formatted.to_csv(os.path.join(region_communal_dir, 'prisons_formatted.csv'), index=False)

    # --- New: Processed Prisons Actual Files ---
    # Filter the processed prisons actual global files by msoa.
    region_age_group = df_age_group_global[df_age_group_global['msoa'].isin(region_msoas)]
    region_age_group.to_csv(os.path.join(region_communal_dir, 'age_group_table.csv'), index=False)

    region_custody_type = df_custody_type_global[df_custody_type_global['msoa'].isin(region_msoas)]
    region_custody_type.to_csv(os.path.join(region_communal_dir, 'custody_type_table.csv'), index=False)

    region_nationality_group = df_nationality_group_global[df_nationality_group_global['msoa'].isin(region_msoas)]
    region_nationality_group.to_csv(os.path.join(region_communal_dir, 'nationality_group_table.csv'), index=False)

    region_offence_group = df_offence_group_global[df_offence_group_global['msoa'].isin(region_msoas)]
    region_offence_group.to_csv(os.path.join(region_communal_dir, 'offence_group_table.csv'), index=False)

    region_ethnicity_group = df_ethnicity_group_global[df_ethnicity_group_global['msoa'].isin(region_msoas)]
    region_ethnicity_group.to_csv(os.path.join(region_communal_dir, 'ethnicity_group_table.csv'), index=False)

    # --- NHS Files ---
    # Create a directory: [...]/input/NHS_trusts/regions/<region>
    region_nhs_dir = os.path.join(nhs_base_dir, 'regions', region)
    os.makedirs(region_nhs_dir, exist_ok=True)

    # Filter NHS files (assuming these files are filtered by msoa).
    region_trusts = df_trusts[df_trusts['msoa'].isin(region_msoas)]
    region_hospitals = df_hospitals[df_hospitals['msoa'].isin(region_msoas)]
    region_hospice = df_hospice[df_hospice['msoa'].isin(region_msoas)]

    # Save NHS files.
    region_trusts.to_csv(os.path.join(region_nhs_dir, 'unique_trust_locations.csv'), index=False)
    region_hospitals.to_csv(os.path.join(region_nhs_dir, 'unique_hospital_locations.csv'), index=False)
    region_hospice.to_csv(os.path.join(region_nhs_dir, 'unique_hospice_locations.csv'), index=False)

    print(f"Region '{region}' data saved:")
    print(f"  Geography -> {region_geo_dir}")
    print(f"  Demography-> {region_demo_dir}")
    print(f"  Communal  -> {region_communal_dir}")
    print(f"  NHS       -> {region_nhs_dir}")

print("\nDone splitting data by region.")
