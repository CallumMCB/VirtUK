# data_loader.py

import os
import geopandas as gpd
import pandas as pd

DATA_PATH = 'data_2021'  # Adjust as needed


class DataLoader:
    def __init__(self, regions=None):
        """
        If regions is provided (as a list or a string), the loader reads data
        from the pre-split region-specific folders. When multiple regions are passed,
        each file is loaded separately and stored in a dictionary keyed by region.
        """
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
        for each region. Otherwise we use the default (un-split) paths.
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
        # Additional communal establishments files:
        self.df_communal_residents_msoa_oa = {}
        self.df_establishment_type_msoa = {}
        self.df_student_accommodation = {}
        self.df_prisons_formatted = {}

        for region, paths_dict in self.file_paths.items():
            # Geography files.
            self.oa_boundaries[region] = gpd.read_file(paths_dict['oa_boundaries_file'])
            self.msoa_boundaries[region] = gpd.read_file(paths_dict['msoa_boundaries_file'])
            self.lad_boundaries[region] = gpd.read_file(paths_dict['lad_boundaries_file'])
            self.df_oa_msoa_lad_regions[region] = pd.read_csv(paths_dict['oa_msoa_lad_regions_file'])
            # Communal establishments files.
            self.df_communal_residents_msoa_oa[region] = pd.read_csv(paths_dict['communal_residents_msoa_oa_file'])
            self.df_care_homes[region] = pd.read_csv(paths_dict['care_homes_file'])
            self.df_resident_type_msoa[region] = pd.read_csv(paths_dict['resident_type_msoa_file'])
            self.df_female_residents[region] = pd.read_csv(paths_dict['female_residents_file'])
            self.df_male_residents[region] = pd.read_csv(paths_dict['male_residents_file'])
            self.df_staff_or_temporary[region] = pd.read_csv(paths_dict['staff_or_temporary_file'])
            # NHS files.
            self.df_trusts[region] = pd.read_csv(paths_dict['trusts_file'])
            self.df_hospitals[region] = pd.read_csv(paths_dict['hospitals_file'])
            self.df_hospices[region] = pd.read_csv(paths_dict['hospice_file'])
            # Demography files.
            self.df_total_age[region] = pd.read_csv(paths_dict['total_age_msoa_file'])
            self.df_female_age[region] = pd.read_csv(paths_dict['female_age_msoa_file'])
            self.df_five_year_ages[region] = pd.read_csv(paths_dict['five_year_ages_file'])
            # Additional communal establishments files.
            self.df_establishment_type_msoa[region] = pd.read_csv(paths_dict['establishment_type_msoa_file'])
            self.df_student_accommodation[region] = pd.read_csv(paths_dict['student_accommodation_file'])
            self.df_prisons_formatted[region] = pd.read_csv(paths_dict['prisons_formatted_file'])

    def process_data(self):
        """
        Process the loaded data. In particular, drop rows with missing coordinates where applicable.
        """
        if self.regions:
            for region in self.regions:
                self.df_care_homes[region] = self.drop_missing_coordinates(
                    self.df_care_homes[region], ['Location Latitude', 'Location Longitude']
                )
                self.df_hospitals[region] = self.drop_missing_coordinates(
                    self.df_hospitals[region], ['latitude', 'longitude']
                )
                self.df_hospices[region] = self.drop_missing_coordinates(
                    self.df_hospices[region], ['latitude', 'longitude']
                )
                self.df_trusts[region] = self.drop_missing_coordinates(
                    self.df_trusts[region], ['latitude', 'longitude']
                )
        else:
            self.df_care_homes = self.drop_missing_coordinates(
                self.df_care_homes, ['Location Latitude', 'Location Longitude']
            )
            self.df_hospitals = self.drop_missing_coordinates(
                self.df_hospitals, ['latitude', 'longitude']
            )
            self.df_hospices = self.drop_missing_coordinates(
                self.df_hospices, ['latitude', 'longitude']
            )
            self.df_trusts = self.drop_missing_coordinates(
                self.df_trusts, ['latitude', 'longitude']
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
                    lad_codes = self.data_loader.df_filtered_hierarchy[region]['lad'].unique()
                    self.data_loader.oa_boundaries[region] = self.filter_dataframe(
                        self.data_loader.oa_boundaries[region], 'OA21CD', oa_codes
                    )
                    self.data_loader.msoa_boundaries[region] = self.filter_dataframe(
                        self.data_loader.msoa_boundaries[region], 'MSOA21CD', msoa_codes
                    )
                    self.data_loader.lad_boundaries[region] = self.filter_dataframe(
                        self.data_loader.lad_boundaries[region], 'LAD24NM', lad_codes
                    )
                    self.data_loader.df_care_homes[region] = self.filter_dataframe(
                        self.data_loader.df_care_homes[region], 'Location Region', [region]
                    )
                    self.data_loader.df_hospitals[region] = self.filter_dataframe(
                        self.data_loader.df_hospitals[region], 'msoa', msoa_codes
                    )
                    self.data_loader.df_hospices[region] = self.filter_dataframe(
                        self.data_loader.df_hospices[region], 'msoa', msoa_codes
                    )
                    self.data_loader.df_trusts[region] = self.filter_dataframe(
                        self.data_loader.df_trusts[region], 'msoa', msoa_codes
                    )
                    # Optionally, you could also filter the new demography and additional communal establishments files,
                    # if needed (depending on how you plan to use them downstream).
            else:
                # In the default (non-region-specific) case, filter the single combined DataFrame.
                self.data_loader.df_filtered_hierarchy = self.data_loader.df_oa_msoa_lad_regions[
                    self.data_loader.df_oa_msoa_lad_regions['region'].isin(self.regions)
                ]
                oa_codes = self.data_loader.df_filtered_hierarchy['area'].unique()
                msoa_codes = self.data_loader.df_filtered_hierarchy['msoa'].unique()
                lad_codes = self.data_loader.df_filtered_hierarchy['lad'].unique()

                self.data_loader.oa_boundaries = self.filter_dataframe(
                    self.data_loader.oa_boundaries, 'OA21CD', oa_codes
                )
                self.data_loader.msoa_boundaries = self.filter_dataframe(
                    self.data_loader.msoa_boundaries, 'MSOA21CD', msoa_codes
                )
                self.data_loader.lad_boundaries = self.filter_dataframe(
                    self.data_loader.lad_boundaries, 'LAD24NM', lad_codes
                )
                self.data_loader.df_care_homes = self.filter_dataframe(
                    self.data_loader.df_care_homes, 'Location Region', self.regions
                )
                self.data_loader.df_hospitals = self.filter_dataframe(
                    self.data_loader.df_hospitals, 'msoa', msoa_codes
                )
                self.data_loader.df_hospices = self.filter_dataframe(
                    self.data_loader.df_hospices, 'msoa', msoa_codes
                )
                self.data_loader.df_trusts = self.filter_dataframe(
                    self.data_loader.df_trusts, 'msoa', msoa_codes
                )

    @staticmethod
    def filter_dataframe(df, column, values):
        return df[df[column].isin(values)].copy()
