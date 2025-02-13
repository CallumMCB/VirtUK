# data_loader.py

import os
import geopandas as gpd
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../data_2021"))

class DataLoader:
    def __init__(self, regions=None):
        # Normalize regions: if a string is passed, wrap it in a list.
        if isinstance(regions, str):
            regions = [regions]
        print(regions)
        self.regions = regions

        self.load_file_paths()
        self.load_data()
        self.process_data()

    def load_file_paths(self):
        """
        Build file paths. If regions are provided, we build a dictionary of file paths
        for each region, otherwise we use the default (un-split) paths.
        """
        # Initialize a dictionary to hold file paths for each region.
        self.file_paths = {}
        for region in self.regions:
            # Construct region-specific directories for each category.
            geo_dir = os.path.join(DATA_PATH, 'input', 'geography', 'regions', region)
            demo_dir = os.path.join(DATA_PATH, 'input', 'demography', 'age_dist_2021')
            com_dir = os.path.join(DATA_PATH, 'input', 'households', 'communal_establishments', 'regions', region)
            nhs_dir = os.path.join(DATA_PATH, 'input', 'NHS_trusts', 'regions', region)

            self.file_paths[region] = {
                # Geography files.
                'oa_boundaries_file': os.path.join(geo_dir, 'oa_boundaries.geojson'),
                'msoa_boundaries_file': os.path.join(geo_dir, 'msoa_boundaries.geojson'),
                'lad_boundaries_file': os.path.join(geo_dir, 'lad_boundaries.geojson'),
                'oa_msoa_lad_regions_file': os.path.join(geo_dir, 'oa_msoa_lad_regions.csv'),
                # Demography files.
                'total_age_msoa_file': os.path.join(demo_dir, 'msoa_total_population.csv'),
                'female_age_msoa_file': os.path.join(demo_dir, 'msoa_female_population.csv'),
                'five_year_ages_file': os.path.join(demo_dir, '5_year_ages_oa_(actual).csv'),
                # Communal establishments files.
                'communal_residents_msoa_oa_file': os.path.join(com_dir, 'communal_residents_msoa_oa.csv'),
                'care_homes_file': os.path.join(com_dir, 'care_homes_occupancy_filled.csv'),
                'resident_type_msoa_file': os.path.join(com_dir, 'resident_type_msoa.csv'),
                'female_residents_file': os.path.join(com_dir, 'female_residents_msoa.csv'),
                'male_residents_file': os.path.join(com_dir, 'male_residents_msoa.csv'),
                'staff_or_temporary_file': os.path.join(com_dir, 'staff_or_temporary_msoa.csv'),
                # Additional communal establishments:
                'establishment_type_msoa_file': os.path.join(com_dir, 'establishment_type_msoa.csv'),
                'student_accommodation_file': os.path.join(com_dir, 'student accommodation.csv'),
                'prisons_formatted_file': os.path.join(com_dir, 'prisons_formatted.csv'),
                # Processed prisons (actual) files (created elsewhere):
                'age_group_file': os.path.join(com_dir, 'age_group_table.csv'),
                'custody_type_file': os.path.join(com_dir, 'custody_type_table.csv'),
                'nationality_group_file': os.path.join(com_dir, 'nationality_group_table.csv'),
                'offence_group_file': os.path.join(com_dir, 'offence_group_table.csv'),
                'ethnicity_group_file': os.path.join(com_dir, 'ethnicity_group_table.csv'),
                # NHS files.
                'trusts_file': os.path.join(nhs_dir, 'unique_trust_locations.csv'),
                'hospitals_file': os.path.join(nhs_dir, 'unique_hospital_locations.csv'),
                'hospice_file': os.path.join(nhs_dir, 'unique_hospice_locations.csv'),
            }

    def load_data(self):
        """
        Load files into GeoDataFrames and DataFrames. When multiple regions are provided,
        each file is loaded and stored in a dictionary keyed by region.
        """
        # Initialize dictionaries for each data category.
        self.oa_boundaries = {}
        self.msoa_boundaries = {}
        self.lad_boundaries = {}
        self.df_oa_msoa_lad_regions = {}
        self.df_care_homes = {}
        self.df_resident_type_msoa = {}
        self.df_female_residents = {}
        self.df_male_residents = {}
        self.df_staff_or_temporary = {}
        self.df_trusts = {}
        self.df_hospitals = {}
        self.df_hospices = {}
        # New demography files:
        self.df_total_age = {}
        self.df_female_age = {}
        self.df_five_year_ages = {}
        # Additional communal establishments files.
        self.df_communal_residents_msoa_oa = {}
        self.df_establishment_type_msoa = {}
        self.df_student_accommodation = {}
        self.df_prisons_formatted = {}
        # Processed prisons actual files.
        self.df_prisons_actual = {}

        # A helper function to load a file given a region, a key, a path, and a loader.
        def load_file(region, key, path, loader):
            try:
                result = loader(path)
                return (region, key, result)
            except Exception as e:
                print(f"Error loading {key} for region {region}: {e}")
                return (region, key, None)

        # Define a mapping from keys to (loader function, target dictionary attribute)
        loaders = {
            'oa_boundaries_file': (gpd.read_file, self.oa_boundaries),
            'msoa_boundaries_file': (gpd.read_file, self.msoa_boundaries),
            'lad_boundaries_file': (gpd.read_file, self.lad_boundaries),
            'oa_msoa_lad_regions_file': (pd.read_csv, self.df_oa_msoa_lad_regions),
            'communal_residents_msoa_oa_file': (pd.read_csv, self.df_communal_residents_msoa_oa),
            'care_homes_file': (pd.read_csv, self.df_care_homes),
            'resident_type_msoa_file': (pd.read_csv, self.df_resident_type_msoa),
            'female_residents_file': (pd.read_csv, self.df_female_residents),
            'male_residents_file': (pd.read_csv, self.df_male_residents),
            'staff_or_temporary_file': (pd.read_csv, self.df_staff_or_temporary),
            'trusts_file': (pd.read_csv, self.df_trusts),
            'hospitals_file': (pd.read_csv, self.df_hospitals),
            'hospice_file': (pd.read_csv, self.df_hospices),
            'total_age_msoa_file': (pd.read_csv, self.df_total_age),
            'female_age_msoa_file': (pd.read_csv, self.df_female_age),
            'five_year_ages_file': (pd.read_csv, self.df_five_year_ages),
            'establishment_type_msoa_file': (pd.read_csv, self.df_establishment_type_msoa),
            'student_accommodation_file': (pd.read_csv, self.df_student_accommodation),
            'prisons_formatted_file': (pd.read_csv, self.df_prisons_formatted),
            # Processed prisons actual files.
            'age_group_file': (pd.read_csv, None),  # Will be handled separately
            'custody_type_file': (pd.read_csv, None),
            'nationality_group_file': (pd.read_csv, None),
            'offence_group_file': (pd.read_csv, None),
            'ethnicity_group_file': (pd.read_csv, None),
        }

        # Use ThreadPoolExecutor to parallelize the loading.
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for region, paths_dict in self.file_paths.items():
                for key, (loader_func, target_dict) in loaders.items():
                    # For the processed prisons actual files, handle them separately:
                    if key in ['age_group_file', 'custody_type_file', 'nationality_group_file',
                               'offence_group_file', 'ethnicity_group_file']:
                        continue
                    path = paths_dict.get(key)
                    if path is not None:
                        futures.append(executor.submit(load_file, region, key, path, loader_func))
            # Wait for all tasks to complete.
            for future in as_completed(futures):
                region, key, result = future.result()
                # Store the result in the corresponding dictionary.
                loaders[key][1][region] = result

        # Now load the processed prisons actual files sequentially (or in parallel similarly)
        # because there are only five files per region.
        self.df_prisons_actual = {}
        for region, paths_dict in self.file_paths.items():
            self.df_prisons_actual[region] = {
                'age_group': pd.read_csv(paths_dict['age_group_file']),
                'custody_type': pd.read_csv(paths_dict['custody_type_file']),
                'nationality_group': pd.read_csv(paths_dict['nationality_group_file']),
                'offence_group': pd.read_csv(paths_dict['offence_group_file']),
                'ethnicity_group': pd.read_csv(paths_dict['ethnicity_group_file']),
            }

    def process_data(self):
        """
        Process the loaded data. In particular, drop rows with missing coordinates where applicable.
        """
        if self.regions:
            for region in self.regions:
                self.df_care_homes[region] = self.drop_missing_coordinates(
                    self.df_care_homes.get(region, pd.DataFrame()), ['Location Latitude', 'Location Longitude']
                )
                self.df_hospitals[region] = self.drop_missing_coordinates(
                    self.df_hospitals.get(region, pd.DataFrame()), ['latitude', 'longitude']
                )
                self.df_hospices[region] = self.drop_missing_coordinates(
                    self.df_hospices.get(region, pd.DataFrame()), ['latitude', 'longitude']
                )

            if isinstance(self.df_care_homes, pd.DataFrame):
                self.df_care_homes = self.drop_missing_coordinates(
                    self.df_care_homes, ['Location Latitude', 'Location Longitude']
                )
            if isinstance(self.df_hospitals, pd.DataFrame):
                self.df_hospitals = self.drop_missing_coordinates(
                    self.df_hospitals, ['latitude', 'longitude']
                )
            if isinstance(self.df_hospices, pd.DataFrame):
                self.df_hospices = self.drop_missing_coordinates(
                    self.df_hospices, ['latitude', 'longitude']
                )

    @staticmethod
    def drop_missing_coordinates(df, coordinate_columns):
        return df.dropna(subset=coordinate_columns).reset_index(drop=True)


class DataFilter:
    def __init__(self, data_loader, regions=None):
        """
        This class filters the data by region. When data is loaded separately per region,
        we iterate over each region. (If using the default, un-split files, we filter the combined data.)
        """
        self.data_loader = data_loader
        self.regions = regions
        self.filter_by_region()

    def filter_by_region(self):
        if self.regions:
            # Check if data is stored per region (i.e. in dictionaries)
            if isinstance(self.data_loader.df_oa_msoa_lad_regions, dict):
                self.data_loader.df_filtered_hierarchy = {}
                for region in self.regions:
                    # Although the files should already be region-specific,
                    # we can still filter by the expected region value.
                    df_hierarchy = self.data_loader.df_oa_msoa_lad_regions[region]
                    self.data_loader.df_filtered_hierarchy[region] = df_hierarchy[
                        df_hierarchy['region'].isin([region])
                    ]
                    oa_codes = self.data_loader.df_filtered_hierarchy[region]['area'].unique()
                    msoa_codes = self.data_loader.df_filtered_hierarchy[region]['msoa'].unique()
                    lad_codes = self.data_loader.df_filtered_hierarchy[region]['lad_codes'].unique()
                    self.data_loader.oa_boundaries[region] = self.filter_dataframe(
                        self.data_loader.oa_boundaries[region], 'OA21CD', oa_codes
                    )
                    self.data_loader.msoa_boundaries[region] = self.filter_dataframe(
                        self.data_loader.msoa_boundaries[region], 'MSOA21CD', msoa_codes
                    )
                    self.data_loader.lad_boundaries[region] = self.filter_dataframe(
                        self.data_loader.lad_boundaries[region], 'LAD24CD', lad_codes
                    )
            else:
                # In the default (non-region-specific) case, filter the single combined DataFrame.
                self.data_loader.df_filtered_hierarchy = self.data_loader.df_oa_msoa_lad_regions[
                    self.data_loader.df_oa_msoa_lad_regions['region'].isin(self.regions)
                ]
                oa_codes = self.data_loader.df_filtered_hierarchy['area'].unique()
                msoa_codes = self.data_loader.df_filtered_hierarchy['msoa'].unique()
                lad_codes = self.data_loader.df_filtered_hierarchy['lad_codes'].unique()

                self.data_loader.oa_boundaries = self.filter_dataframe(
                    self.data_loader.oa_boundaries, 'OA21CD', oa_codes
                )
                self.data_loader.msoa_boundaries = self.filter_dataframe(
                    self.data_loader.msoa_boundaries, 'MSOA21CD', msoa_codes
                )
                self.data_loader.lad_boundaries = self.filter_dataframe(
                    self.data_loader.lad_boundaries, 'LAD24CD', lad_codes
                )

    @staticmethod
    def filter_dataframe(df, column, values):
        return df[df[column].isin(values)].copy()
