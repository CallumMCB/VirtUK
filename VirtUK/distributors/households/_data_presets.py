from dataclasses import dataclass
import numpy as np
import pandas as pd
import yaml
import pickle

from VirtUK import paths

@dataclass(frozen=True)
class FilePaths:
    # Geography
    geography_dir = f'{paths.data_path}/input/geography/'

    hierarchy_fp = (geography_dir + 'oa_msoa_lad_regions.csv') # Not sure if this will be needed but no harm in importing

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
    oa_student_accom = (communal_dir + 'student_accommodation.csv')
    msoa_establishment_type = (communal_dir + 'establishment_type_msoa.csv')
    msoa_resident_type = (communal_dir + 'resident_type_msoa.csv')
    msoa_staff_or_temporary = (communal_dir + 'staff_or_temporary_msoa.csv')
    msoa_female_residents = (communal_dir + 'female_residents_msoa.csv')
    msoa_male_residents = (communal_dir + 'male_residents_msoa.csv')
    care_homes = (communal_dir + 'care_homes_beds_type_locations.csv')

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
    def __init__(
            self,
            file_paths: FilePaths = FilePaths(),
    ):
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
            'collections':{
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

        return {
            'students': self.load_oa_student_accommodation(),
            'total': tot_df
        }

    def load_oa_student_accommodation(self):
        file_path = self.file_paths.oa_student_accom

        # Load the processed data
        df = pd.read_csv(file_path, dtype={col: 'uint16' for col in pd.read_csv(file_path, nrows=1).columns if
                                             col not in ['area', 'accommodation type']})

        # Set multi-index by 'areas' and 'accommodation type'
        df.set_index(['area', 'accommodation type'], inplace=True)

        return df

    def load_msoa_communal_data(self):
        return {
            'communal_type':{
                'establishment_type': self.load_csv16(self.file_paths.msoa_establishment_type, index_cols=['msoa']),
                'resident_type': self.load_csv16(self.file_paths.msoa_resident_type, index_cols=['msoa']),
            },
            'staff_or_temporary_residents': self.load_csv16(self.file_paths.msoa_staff_or_temporary, index_cols=['msoa']),
            'residents_by_sex':{
                'female_residents': self.load_csv16(self.file_paths.msoa_female_residents, index_cols=['msoa']),
                'male_residents': self.load_csv16(self.file_paths.msoa_male_residents, index_cols=['msoa']),
            }
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
        # Load the CSV file
        df = pd.read_csv(self.file_paths.lad_dependent_children_ratios)

        # Set Local authority code as the primary index and drop the 'Local authority code (2022)' column
        df.set_index('Local authority name (2022)', inplace=True)
        df.drop(columns=['Local authority code (2022)'], inplace=True)

        # Rename the dependent children column categories
        df['Number of dependent children living in the household'] = df[
            'Number of dependent children living in the household'].replace({
            'No dependent children': '0',
            'One dependent child': '1',
            'Two dependent children': '2',
            'Three or more dependent children': '3'
        })
        df.rename(columns={'Number of dependent children living in the household': 'Number of kids'}, inplace=True)

        # Drop the row where 'Number of kids' is '0'
        df = df[df['Number of kids'] != '0']

        # Merge lone parent columns into "1 parent"
        lone_parent_cols = [
            'Single family household: Lone parent (female)',
            'Single family household: Lone parent (male)'
        ]
        df['1 parent'] = df[lone_parent_cols].sum(axis=1)

        # Merge couple columns into "2 parents"
        couple_cols = [
            'Single family household: Non-step couple',
            'Single family household: Step couple'
        ]
        df['2 parents'] = df[couple_cols].sum(axis=1)

        # Drop the original lone parent and couple columns along with "Sum"
        df.drop(columns=lone_parent_cols + couple_cols + ['Sum'], inplace=True)

        # Renormalize the values for '1 parent' and '2 parents' columns for each local authority individually
        def renormalize(group):
            columns_to_renormalize = ['1 parent', '2 parents']
            total_sum = group[columns_to_renormalize].sum().sum()
            return group[columns_to_renormalize].div(total_sum) if total_sum != 0 else group[columns_to_renormalize]

        # Apply the renormalization function to each group
        df[['1 parent', '2 parents']] = df.groupby(level=0).apply(lambda group: renormalize(group)).values

        # Set the renamed dependent children column as the secondary index
        df.set_index('Number of kids', append=True, inplace=True)

        return df

    def load_all_data(self):
        return {
            'ac': self.load_age_counts(),
            'hrcy': self.load_hierarchy(),
            'hh_comps': self.load_household_compositions(),
            'hh_sizes': self.load_household_sizes(),
            'oa_cd': self.load_oa_communal_data(),
            'msoa_cd': self.load_msoa_communal_data(),
            'msoa_cpad': self.load_child_parent_age_dist_msoa(),
            'cads': self.load_couples_age_disparities(),
            'oa_ptnr_status': self.load_partnership_status_oa(),
            'lad_ptnr_status': self.load_partnership_status_lad(),
            'hct': self.load_child_parent_age_dist_msoa(),
            'lad_dc': self.load_dependent_children()
        }

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