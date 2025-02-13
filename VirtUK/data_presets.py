from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml
import pickle

from VirtUK import paths

@dataclass(frozen=True)
class FilePaths:
    # Geography
    geography_dir = f'{paths.data_path}/input/geography/'

    hierarchy_fp = (geography_dir + 'oa_msoa_lad_regions.csv')

    # Households
    household_dir =  f'{paths.data_path}/input/households/'

    household_size_fp = (household_dir + 'household_size.csv')
    household_composition_fp = (household_dir + 'household_composition.csv')
    same_sex_age_disparity = (household_dir + 'same_sex_age_disparity.csv')
    hetero_age_disparity = (household_dir + 'age_difference_interpolations.csv')
    child_parent_age_dists_fp = (household_dir + 'fitted_child_parent.npy')
    lad_dependent_children_ratios = (household_dir + 'dependent_children_ratios.csv')

    household_comp_config_fp = f'{paths.configs_path}/defaults/distributors/household_distributor_2021.yaml'

    # Communal
    communal_dir = (household_dir + 'communal_establishments/')

    oa_communal_residents_fp = (communal_dir + 'communal_residents_msoa_oa.csv')
    oa_student_accom = (communal_dir + 'oa_student_accommodation.csv')
    msoa_student_accom = (communal_dir + 'msoa_student_accommodation.csv')
    msoa_establishment_type = (communal_dir + 'establishment_type_msoa.csv')
    msoa_resident_type = (communal_dir + 'resident_type_msoa.csv')
    msoa_staff_or_temporary = (communal_dir + 'staff_or_temporary_msoa.csv')
    msoa_female_residents = (communal_dir + 'female_residents_msoa.csv')
    msoa_male_residents = (communal_dir + 'male_residents_msoa.csv')
    care_homes = (communal_dir + 'care_homes/' + 'occupancy_filled.csv')
    prisons = (communal_dir + 'prisons/' + 'prisons_formatted.csv')

    # Demographics
    demography_dir = f'{paths.data_path}/input/demography/'

    age_dir = (demography_dir + 'age_dist_2021/')

    # 1 year ages are less accurate and should be used with caution!
    ages_total = (age_dir + '1_year_ages_total_oa.csv')
    ages_male = (age_dir + '1_year_ages_male_oa.csv')
    ages_female = (age_dir + '1_year_ages_female_oa.csv')

    broad_ages = (age_dir + 'age_broad_bands.csv')

    students_over_5 = (demography_dir + 'number_students_over_5.csv')

    partnership_status_dir = (demography_dir + 'partnership_status/')

    lad_partnership_5y_interpolated = (partnership_status_dir + 'lad/all_partnership_interpolations.pkl')
    oa_partnership_status = (partnership_status_dir + 'oa_legal_partnership_simplified.pkl')

class DataLoader:
    def __init__(self, file_paths: FilePaths = FilePaths()):
        self.file_paths = file_paths

    @staticmethod
    def load_csv16(filename, index_cols):
        # Read a sample row to determine column types
        sample_df = pd.read_csv(filename, nrows=1)
        # Create a dtype dictionary excluding index columns
        dtype_dict = {column: 'uint16' for column in sample_df.columns if column not in index_cols}
        # Load the DataFrame with specified dtypes
        df = pd.read_csv(filename, dtype=dtype_dict)
        # Ensure index columns retain their original types as strings
        for col in index_cols:
            df[col] = df[col].astype(str)
        # Set the hierarchical index
        df.set_index(index_cols, inplace=True)
        return df

    @staticmethod
    def load_csv64(filename):
        sample_df = pd.read_csv(filename, nrows=1, index_col=0)
        dtype_dict = {column: 'int64' for column in sample_df.columns}
        return pd.read_csv(filename, index_col=0, dtype=dtype_dict)

    def load_age_counts(self):
        ages_total = self.load_csv16(self.file_paths.ages_total, index_cols=['area'])
        ages_male = self.load_csv16(self.file_paths.ages_male, index_cols=['area'])
        ages_female = self.load_csv16(self.file_paths.ages_female, index_cols=['area'])
        ages_broad = self.load_csv64(self.file_paths.broad_ages)
        return {
            'total': ages_total,
            'm': ages_male,
            'f': ages_female,
            'broad': ages_broad
        }

    def load_hierarchy(self):
        return pd.read_csv(self.file_paths.hierarchy_fp)

    def load_household_comp_config(self):
        with open(self.file_paths.household_comp_config_fp, 'r') as f:
            return yaml.safe_load(f)['default_household_compositions']

    def load_household_compositions(self):
        hh_comps = self.load_household_comp_config()
        old_hh_comps = {key: value for key, value in hh_comps.items() if value.get('household') == 'old'}
        nd_family_hh_comps = {key: value for key, value in hh_comps.items() if value.get('household') == 'nd_family'}
        d_family_hh_comps = {key: value for key, value in hh_comps.items() if value.get('household') == 'd_family'}
        no_kids_hh_comps = {key: value for key, value in hh_comps.items() if value.get('household') == 'nokids'}
        other_hh_comps = {key: value for key, value in hh_comps.items() if value.get('household') == 'other'}

        hh_numbers = pd.read_csv(self.file_paths.household_composition_fp, index_col=0)
        return {
            'config': hh_comps,
            'numbers': hh_numbers,
            'collections': {
                'old': old_hh_comps,
                'nd_family': nd_family_hh_comps,
                'd_family': d_family_hh_comps,
                'no_kids': no_kids_hh_comps,
                'other': other_hh_comps
            }
        }

    def load_household_sizes(self):
        return self.load_csv16(self.file_paths.household_size_fp, index_cols=['area'])

    def load_oa_communal_data(self):
        tot_df = self.load_csv16(self.file_paths.oa_communal_residents_fp, index_cols=['msoa', 'area'])
        tot_df.drop(columns=['household residents'], inplace=True)
        tot_df.rename(columns={'communal residents': 'numbers'}, inplace=True)
        tot_df = tot_df[tot_df['numbers'] > 0]
        return tot_df

    def load_oa_student_accommodation(self):
        file_path = self.file_paths.oa_student_accom
        # Load the processed data
        df = pd.read_csv(
            file_path,
            dtype={col: 'uint16' for col in pd.read_csv(file_path, nrows=1).columns if col not in ['area', 'accommodation type']}
        )
        # Set multi-index by 'area' and 'accommodation type'
        df.set_index(['area', 'accommodation type'], inplace=True)
        return df

    def load_msoa_student_accommodation(self):
        file_path = self.file_paths.msoa_student_accom
        # Load the processed data
        df = pd.read_csv(
            file_path,
            dtype={col: 'uint16' for col in pd.read_csv(file_path, nrows=1).columns if col not in ['msoa', 'accommodation type']}
        )
        # Set multi-index by 'msoa' and 'accommodation type'
        df.set_index(['msoa', 'accommodation type'], inplace=True)
        return df

    def load_care_homes_cqc(self):
        cqc_df = pd.read_csv(self.file_paths.care_homes)
        CQC_columns = {
            "Location Name": "Name",
            "Location Postal Code": "Postcode",
            "area": "area",
            "msoa": "msoa",
            "Service type - Care home service with nursing": "Nursing",
            "Service type - Care home service without nursing": "Non-Nursing",
            "Care homes beds": "Beds",
            "occupancy_mean": "Mean Occupancy",
            "occupancy_std": "Occupancy std.",
            "Service user band - Children 0-18 years": "Children",
            "Service user band - Younger Adults": "Youger Adults",
            "Service user band - Older People": "Older Adults"
        }
        cqc_df = cqc_df[list(CQC_columns.keys())]
        cqc_df = cqc_df.rename(columns=CQC_columns)
        cqc_df.set_index('msoa', inplace=True)
        return cqc_df

    def load_prisons(self):
        prisons_df = pd.read_csv(self.file_paths.prisons)
        prisons_df.set_index('msoa', inplace=True)
        return prisons_df

    def load_msoa_communal_data(self):
        # --- Merge communal type files ---
        establishment_type_df = self.load_csv16(self.file_paths.msoa_establishment_type, index_cols=['msoa'])
        resident_type_df = self.load_csv16(self.file_paths.msoa_resident_type, index_cols=['msoa'])
        establishment_type_df['communal_type'] = 'establishment_type'
        resident_type_df['communal_type'] = 'resident_type'
        establishment_type_df = establishment_type_df.set_index('communal_type', append=True)
        resident_type_df = resident_type_df.set_index('communal_type', append=True)
        communal_type_df = pd.concat([establishment_type_df, resident_type_df]).sort_index()
        # --- Load staff or temporary residents ---
        staff_or_temporary_residents_df = self.load_csv16(self.file_paths.msoa_staff_or_temporary, index_cols=['msoa'])
        # --- Merge residents by sex files ---
        female_df = self.load_csv16(self.file_paths.msoa_female_residents, index_cols=['msoa'])
        male_df = self.load_csv16(self.file_paths.msoa_male_residents, index_cols=['msoa'])
        female_df['sex'] = 'female'
        male_df['sex'] = 'male'
        female_df = female_df.set_index('sex', append=True)
        male_df = male_df.set_index('sex', append=True)
        residents_by_sex_df = pd.concat([female_df, male_df]).sort_index()
        return {
            'communal_type': communal_type_df,
            'staff_or_temporary_residents': staff_or_temporary_residents_df,
            'residents_by_sex': residents_by_sex_df
        }

    def load_couples_age_disparities(self):
        return {
            'hetero': pd.read_csv(self.file_paths.hetero_age_disparity, index_col=0).rename(columns=lambda x: int(x)),
            'same_sex': pd.read_csv(self.file_paths.same_sex_age_disparity).to_dict(orient='dict')
        }

    def load_child_parent_age_dist_msoa(self):
        return np.load(self.file_paths.child_parent_age_dists_fp, allow_pickle=True).item()

    def load_partnership_status_oa(self):
        with open(self.file_paths.oa_partnership_status, 'rb') as f:
            return pickle.load(f)

    def load_partnership_status_lad(self):
        with open(self.file_paths.lad_partnership_5y_interpolated, 'rb') as f:
            return pickle.load(f)

    def load_dependent_children(self):
        df = pd.read_csv(self.file_paths.lad_dependent_children_ratios)
        df.set_index('Local authority name (2022)', inplace=True)
        df.drop(columns=['Local authority code (2022)'], inplace=True)
        df['Number of dependent children living in the household'] = df[
            'Number of dependent children living in the household'
        ].replace({
            'No dependent children': '0',
            'One dependent child': '1',
            'Two dependent children': '2',
            'Three or more dependent children': '3'
        })
        df.rename(
            columns={'Number of dependent children living in the household': 'Number of kids'},
            inplace=True
        )
        df = df[df['Number of kids'] != '0']
        lone_parent_cols = [
            'Single family household: Lone parent (female)',
            'Single family household: Lone parent (male)'
        ]
        df['1 parent'] = df[lone_parent_cols].sum(axis=1)
        couple_cols = [
            'Single family household: Non-step couple',
            'Single family household: Step couple'
        ]
        df['2 parents'] = df[couple_cols].sum(axis=1)
        df.drop(columns=lone_parent_cols + couple_cols + ['Sum'], inplace=True)
        def renormalize(group):
            columns_to_renormalize = ['1 parent', '2 parents']
            total_sum = group[columns_to_renormalize].sum().sum()
            return group[columns_to_renormalize].div(total_sum) if total_sum != 0 else group[columns_to_renormalize]
        df[['1 parent', '2 parents']] = df.groupby(level=0).apply(lambda group: renormalize(group)).values
        df.set_index('Number of kids', append=True, inplace=True)
        return df

    def load_all_data(self):
        """
        Load all datasets in parallel using ThreadPoolExecutor.
        Returns a dictionary with keys corresponding to the data names.
        """
        tasks = {
            'ac': self.load_age_counts,
            'hrcy': self.load_hierarchy,
            'hh_comps': self.load_household_compositions,
            'hh_sizes': self.load_household_sizes,
            'oa_cd': self.load_oa_communal_data,
            'msoa_cd': self.load_msoa_communal_data,
            'oa_stdnts': self.load_oa_student_accommodation,
            'msoa_stdnts': self.load_msoa_student_accommodation,
            'cqc_chs': self.load_care_homes_cqc,
            'prisons': self.load_prisons,
            'msoa_cpad': self.load_child_parent_age_dist_msoa,
            'cads': self.load_couples_age_disparities,
            'oa_ptnr_status': self.load_partnership_status_oa,
            'lad_ptnr_status': self.load_partnership_status_lad,
            'hct': self.load_child_parent_age_dist_msoa,  # If duplicate, consider removing one.
            'lad_dc': self.load_dependent_children
        }

        results = {}
        with ThreadPoolExecutor() as executor:
            # Submit all tasks concurrently.
            future_to_key = {executor.submit(func): key for key, func in tasks.items()}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    print(f"Error loading {key}: {e}")
        return results


@dataclass(frozen=True)
class AgeLimits:
    kid_max_age: int = 15

    student_min_age: int = 18
    student_max_age: int = 25
    old_min_age: int = 65
    old_max_age: int = 99
    adult_min_age: int = 18
    adult_max_age: int = 64
    young_adult_min_age: int = 18
    young_adult_max_age: int = 24
    max_age_to_be_parent: int = 64