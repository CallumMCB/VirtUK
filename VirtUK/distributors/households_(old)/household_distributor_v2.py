from pprint import pprint
from tabulate import tabulate

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import logging

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import yaml
import pickle

from VirtUK import paths
from VirtUK.demography import Person, person
from VirtUK.geography import Geography, Area, SuperArea, LAD, Region
from VirtUK.groups import Household, Households
from VirtUK.groups import Temp_OA_Households

logger = logging.getLogger("household_distributor")

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

    oa_communal_residents_fp = (communal_dir + 'communal_residents_oa.csv')
    msoa_establishment_type = (communal_dir + 'establishment_type_msoa.csv')
    msoa_resident_type = (communal_dir + 'resident_type_msoa.csv')
    msoa_staff_or_temporary = (communal_dir + 'staff_or_temporary_msoa.csv')
    msoa_female_residents = (communal_dir + 'female_residents_msoa.csv')
    msoa_male_residents = (communal_dir + 'male_residents_msoa.csv')

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
    def load_csv16(filename):
        sample_df = pd.read_csv(filename, nrows=1, index_col=0)
        dtype_dict = {column: 'uint16' for column in sample_df.columns}
        return pd.read_csv(filename, index_col=0, dtype=dtype_dict)

    def load_age_counts(self):
        ages_total = self.load_csv16(self.file_paths.ages_total)
        ages_male = self.load_csv16(self.file_paths.ages_male)
        ages_female = self.load_csv16(self.file_paths.ages_female)
        ages_broad = self.load_csv16(self.file_paths.broad_ages)

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
        return self.load_csv16(self.file_paths.household_size_fp)

    def load_oa_communal_residents(self):
        df = self.load_csv16(self.file_paths.oa_communal_residents_fp)
        df.drop(columns=['household residents'], inplace=True)
        df.rename(columns={'communal residents': 'numbers'}, inplace=True)
        df = df[df['numbers'] > 0]
        return df

    def load_msoa_communal_data(self):
        return {
            'communal_type':{
                'establishment_type': self.load_csv16(self.file_paths.msoa_establishment_type),
                'resident_type': self.load_csv16(self.file_paths.msoa_resident_type),
            },
            'staff_or_temporary_residents': self.load_csv16(self.file_paths.msoa_staff_or_temporary),
            'residents_by_sex':{
                'female_residents': self.load_csv16(self.file_paths.msoa_female_residents),
                'male_residents': self.load_csv16(self.file_paths.msoa_male_residents),
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
            'oa_cr': self.load_oa_communal_residents(),
            'msoa_cd': self.load_msoa_communal_data(),
            'msoa_cpad': self.load_child_parent_age_dist_msoa(),
            'cads': self.load_couples_age_disparities(),
            'oa_ptnr_status': self.load_partnership_status_oa(),
            'lad_ptnr_status': self.load_partnership_status_lad(),
            'hct': self.load_child_parent_age_dist_msoa(),
            'lad_dc': self.load_dependent_children()
        }

class HouseholdContext:
    def __init__(self):
        self.area = None
        self.all_households = []
        self.women_by_age = defaultdict(list)
        self.men_by_age = defaultdict(list)
        self.number_households = pd.DataFrame
        self.size_households = np.ndarray([])
        self.partnerships_oa = pd.DataFrame()
        self.partnerships_lad = pd.DataFrame()
        self.dependent_kids_lad = pd.DataFrame()

class HouseholdDistributor:
    def __init__(
            self,
            age_counts: Dict[str, pd.DataFrame],

            hierarchy: pd.DataFrame,

            household_comp_config: Dict[str, Any],
            household_comp_numbers: pd.DataFrame,
            household_comp_collections: Dict[str, pd.DataFrame],
            household_sizes: pd.DataFrame,

            oa_communal_residents: pd.DataFrame,
            msoa_communal_residents: Dict[str, pd.DataFrame],
            msoa_communal_types: Dict[str, pd.DataFrame],
            msoa_staff_temporary_residents: pd.DataFrame,

            child_parent_age: Dict[str, Dict[int, List[Tuple[np.ndarray, np.ndarray]]]],
            couples_age_disparity: Dict[str, Any],
            oa_partner_status: pd.DataFrame,
            lad_partner_status: pd.DataFrame,
            lad_dependent_children: pd.DataFrame,

            age_limits: AgeLimits = AgeLimits(),
    ):
        self.age_counts = age_counts

        self.hierarchy = hierarchy

        self.household_comp_config = household_comp_config
        self.config_size_households = {
            'fixed': {key:
                          {'size': values['size'],
                           'comp': np.array(values['composition']),
                           'type': values['household']}
                      for key, values in household_comp_config.items() if isinstance(values['size'], int)},
            'range': {key:
                          {'size': int(np.floor(values['size'])),
                           'comp': np.array(values['composition']),
                           'type': values['household']}
                      for key, values in household_comp_config.items() if isinstance(values['size'], float)}
        }
        self.household_comp_numbers = household_comp_numbers
        self.household_comp_collections = household_comp_collections
        self.household_sizes = household_sizes
        self.oa_communal_residents = oa_communal_residents
        self.msoa_communal_residents = msoa_communal_residents
        self.msoa_communal_types = msoa_communal_types
        self.msoa_staff_temporary_residents = msoa_staff_temporary_residents

        self.child_parent_age = child_parent_age
        self.couples_age_disparity = couples_age_disparity
        self.oa_partner_status = oa_partner_status
        self.lad_partner_status = lad_partner_status
        self.lad_dependent_children = lad_dependent_children

        self.age_limits = age_limits

    @classmethod
    def from_data_loader(
            cls,
            data_loader: DataLoader = DataLoader(),
            age_limits: AgeLimits = AgeLimits(),
    )-> "HouseholdDistributor":
        all_data = data_loader.load_all_data()
        return cls(
            age_counts = all_data['ac'],
            hierarchy = all_data['hrcy'],
            household_comp_config = all_data['hh_comps']['config'],
            household_comp_numbers = all_data['hh_comps']['numbers'],
            household_comp_collections = all_data['hh_comps']['collections'],
            oa_communal_residents = all_data['oa_cr'],
            msoa_communal_residents = all_data['msoa_cd']['residents_by_sex'],
            msoa_communal_types = all_data['msoa_cd']['communal_type'],
            msoa_staff_temporary_residents = all_data['msoa_cd']['staff_or_temporary_residents'],
            household_sizes = all_data['hh_sizes'],
            child_parent_age = all_data['msoa_cpad'],
            couples_age_disparity = all_data['cads'],
            oa_partner_status= all_data['oa_ptnr_status'],
            lad_partner_status = all_data['lad_ptnr_status'],
            lad_dependent_children = all_data['lad_dc'],
            age_limits = age_limits
        )

    def distribute_people_and_households_to_areas(
            self,
            lads: List[LAD]
    ):
        logger.info("Distributing people to households")

        area_names = [area.name for lad in lads for area in lad.areas]

        household_numbers_df = self.household_comp_numbers.loc[area_names]
        household_sizes_df = self.household_sizes.loc[area_names]
        oa_partnership_status_df = self.oa_partner_status.loc[area_names]

        for lad in lads:
            partnerships_lad = self.lad_partner_status.loc[lad.name]
            dependent_kids_lad = self.lad_dependent_children.loc[lad.name]
            for counter, (area,
                          (_, number_households),
                          (_, size_households)) in (
                    tqdm(enumerate(zip(
                        lad.areas,
                        household_numbers_df.iterrows(),
                        household_sizes_df.iterrows(),
                    ), start=1),
                total = len(lad.areas),
                desc="Distributing people to households"
            )):

                ctxt = HouseholdContext()

                ctxt.area = area
                ctxt.all_households = []
                ctxt.number_households = number_households
                ctxt.size_households = size_households.to_numpy()
                ctxt.partnerships_oa = oa_partnership_status_df.loc[area.name]
                ctxt.partnerships_lad = partnerships_lad
                ctxt.dependent_kids_lad = dependent_kids_lad

                self.ppl = People(ctxt, self)

                area.households = self.distribute_people_to_households(ctxt)

    def distribute_people_to_households(self, ctxt)-> Households:
        collection_distributors = {
            'old': OldPeopleHousehold,
            'nokids': NoKidsHousehold,
            'd_family': DependentHousehold,
            'nd_family': NonDependentHousehold,
            'other': OtherHousehold,
        }

        for collection, distributor in collection_distributors.items():
            sub_collections = self.household_comp_collections[collection].keys()
            distributor(self).create_households(sub_collections, ctxt)
        exit()

    def _create_new_household(
            self,
            area: Area,
            type: str,
            initial_occupants: Optional[List[Person]] = None,
            max_household_size: int = np.inf
    )->Household:
        household = Household(
            type = type,
            max_size = max_household_size,
            area = area,
        )
        for initial_occupant in initial_occupants:
            household.add(initial_occupant)
        return household


    def select_random_household_occupants(self, potential_households, number_households):
        # Select a number of items (households) from the 2D list without replacement
        if number_households > len(potential_households):
            raise ValueError("number_households cannot be greater than the number of available households")

        selected_indices = set(random.sample(range(len(potential_households)), number_households))
        selected_occupants = [potential_households[i] for i in selected_indices]

        # Remove the selected households from the original list efficiently
        potential_households[:] = [household for i, household in enumerate(potential_households) if
                                  i not in selected_indices]

        return selected_occupants, potential_households

class People(HouseholdDistributor):
    def __init__(self, context: HouseholdContext, distributor: HouseholdDistributor):
        self.ctxt = context
        self.distributor = distributor

        self._create_people_dicts(self.ctxt.area)

        self.assignment = self.Assignment(self)
        self.partnership = self.Partnership(self)
        self.kids = self.Kids(self)
        self.people_checker = self.PeopleChecker(self)

    def _create_people_dicts(self, area: Area):
        """
        Creates dictionaries with the men and women per age key living in the area.
        """
        self.ctxt.men_by_age.clear()
        self.ctxt.women_by_age.clear()

        grouped = defaultdict(list)
        men, women = 0, 0
        for person in filter(lambda p: p.residence is None, area.people):
            if person.age >= 0:
                if person.sex == 'm':
                    men += 1
                elif person.sex =='f':
                    women += 1
            grouped[(person.sex, person.age)].append(person)

        for (sex, age), people in grouped.items():
            if sex == 'm':
                self.ctxt.men_by_age[age].extend(people)
            elif sex == 'f':
                self.ctxt.women_by_age[age].extend(people)
            else:
                print(f"Unexpected sex value: {sex}")

    def _remove_from_people_dict(self, people_dict, people_to_remove):
        removal_map = defaultdict(list)
        for person in people_to_remove:
            removal_map[person.age].append(person)

        for age, persons in removal_map.items():
            people_dict[age] = [p for p in people_dict[age] if p not in persons]
            if not people_dict[age]:
                del people_dict[age]

    class Assignment:
        def __init__(self, people_instance):
            self.ppl_inst = people_instance
            self.ctxt = self.ppl_inst.ctxt
            print(f"\n___ Output Area {self.ctxt.area.name} ___\n")
            print("\n=== Population Breakdown ===")
            self.smrzd_pop, print_statement = self.summarize_population()
            print(print_statement)

            print("\n=== Household Information ===")
            print("\nSize of Households:")
            self.size_hhs = pd.concat([
                    pd.DataFrame({"Size of Household": list(range(1, len(self.ctxt.size_households))),
                                  "Number of Households": self.ctxt.size_households[:-1]}),
                    pd.DataFrame({"Size of Household": ["Total"],
                                  "Number of Households": [self.ctxt.size_households[-1]]})
                ]).set_index("Size of Household")
            print(self.size_hhs.T)
            print("\nNumber of Households:")
            self.available_hhs = self.ctxt.number_households[self.ctxt.number_households.values != 0][:-1]
            print(tabulate(pd.DataFrame(self.available_hhs), headers='keys', tablefmt = 'pipe'))

            base_config = self.base_configuration()

        def summarize_population(self):
            def summarize_group(group_by_age):
                old = sum(len(lst) for age, lst in group_by_age.items() if age >= 65)
                adult = sum(len(lst) for age, lst in group_by_age.items() if 22 <= age < 65)
                ya = sum(len(lst) for age, lst in group_by_age.items() if 17 <= age < 22)
                kid = sum(len(lst) for age, lst in group_by_age.items() if 0 <= age < 17)
                return np.array([kid, ya, adult, old]), {
                    "Old": old,
                    "Adult": adult,
                    "Young Adult": ya,
                    "Kid": kid
                }
            w_array, women_summary = summarize_group(self.ctxt.women_by_age)
            m_array, men_summary = summarize_group(self.ctxt.men_by_age)

            summary_output = "\nWomen:\n"
            for category, count in women_summary.items():
                summary_output += f"  {category}: {count}\n"

            summary_output += "\nMen:\n"
            for category, count in men_summary.items():
                summary_output += f"  {category}: {count}\n"

            return w_array + m_array, summary_output

        def base_configuration(self):
            print("\n=== Assignment of Fixed Households ===")
            print("\nAvailable People:")
            print(pd.DataFrame(np.reshape(self.smrzd_pop, (-1, 4)), columns = ['Kids', 'YA', 'Adults', 'Old']).to_string(index=False))
            fixed_hhs = {
                key: values
                for key, values in self.ppl_inst.distributor.config_size_households['fixed'].items()
                if key in self.available_hhs.index
            }

            overall_target = np.array([0] + list(self.size_hhs[self.size_hhs.index != 'Total']['Number of Households']))

            print("\nFixed Households:")
            pprint(fixed_hhs)
            for key, values in fixed_hhs.items():
                number_hhs = self.available_hhs[key]
                self.smrzd_pop -= number_hhs * values['comp']
                self.size_hhs.loc[values['size']] -= number_hhs

            print("\nRemaining Available People:")
            print(pd.DataFrame(np.reshape(self.smrzd_pop, (-1, 4)), columns = ['Kids', 'YA', 'Adults', 'Old']).to_string(index=False))
            print("\nRemaining Sizes of Households:")
            print(self.size_hhs.T)

            print("\n##### Running Temp OA Households ######\n")
            fixed_comp_number = {
                key: {'comp': values['comp'], 'size': values['size'], 'type': values['type'], 'number': self.available_hhs[key]}
                for key, values in self.ppl_inst.distributor.config_size_households['fixed'].items()
                if key in self.available_hhs.index
            }
            variable_comp_number = {
                key: {'comp': values['comp'], 'size': values['size'], 'type': values['type'], 'number': self.available_hhs[key]}
                for key, values in self.ppl_inst.distributor.config_size_households['range'].items()
                if key in self.available_hhs.index
            }

            var_target = np.array([0] + list(self.size_hhs[self.size_hhs.index != 'Total']['Number of Households']))

            base_temp_oa = Temp_OA_Households(
                area=self.ctxt.area,
                overall_target_dist=overall_target,
                fixed_hh_composition=fixed_comp_number,
                var_hh_composition=variable_comp_number,
                person_counts=self.smrzd_pop,
                var_target_dist=var_target,
                dependent_kids_ratios=self.ctxt.dependent_kids_lad,
                num_simple_family_hhs=100
            )
            base_temp_oa.initialize()

            print("\n--- Fixed Households ---\n")
            base_temp_oa._print_grouped_households_table(base_temp_oa.fixed_households)

            # Specify the number of branches you want
            num_branches = 1
            # Use a list comprehension to create the specified number of branches
            branches = [base_temp_oa.branch() for _ in range(num_branches)]

            for i, branch in enumerate(branches):
                print("\n" + "=" * 200)
                print(f"{' RUNNING BRANCH '.center(200, '=')}")
                print(f" Branch {i + 1} ".center(200, '='))
                print("=" * 200 + "\n")
                branch.run()

            print('continue base')
            exit()

    class Partnership:
        def __init__(self, people_instance):
            self.people_instance = people_instance

        def _status_for_age_bracket(self, age_band: str, partnership_status: str):
            occupants = []
            if partnership_status == 'partnered':
                occupants.extend(self._find_and_set_partner(age_band, 'f'))
            else: occupants.extend(self._assign_individual_status(age_band))
            print(self.ctxt.size_households)
            print(f"Women after {partnership_status} process: {sum(len(lst) for age, lst in self.ctxt.women_by_age.items() if age >= 65)}")
            print(f"Men after {partnership_status} process: {sum(len(lst) for age, lst in self.ctxt.men_by_age.items() if age >= 65)}")

            return occupants

        def _assign_individual_status(self, age_band):
            low_age, high_age = map(int, age_band.split('-'))

            # Define possible statuses
            statuses = ['divorced_separated', 'never_partnered', 'widowed']
            all_people_assigned = {'m': [], 'f': []}
            all_individuals = []
            # Loop through both sexes
            for sex in ['f', 'm']:
                # Retrieve the dictionary for the given sex
                people_dict = self.ctxt.women_by_age if sex == 'f' else self.ctxt.men_by_age

                # Loop through each age within the given age range
                for age in filter(lambda x: low_age <= x <= high_age, people_dict.keys()):
                    # Extract all people at this age
                    people_at_age = people_dict[age]

                    # Extract raw probabilities for each status for the given sex and age
                    raw_probabilities = [
                        self.ctxt.partnerships_lad.loc[(sex, status), age] for status in statuses
                    ]

                    normalized_probabilities = raw_probabilities/sum(raw_probabilities)

                    # Assign status to each individual at this age
                    for person in people_at_age:
                        assigned_status = np.random.choice(statuses, p=normalized_probabilities)
                        person.partnership_status = assigned_status
                        all_people_assigned[sex].append(person)
                        all_individuals.append([person])

                # print("___________")
                # print(f"People of sex {sex}:")
                # for status in statuses:
                #     print(f"Status: {status}")
                #     people_in_status = [person for person in all_people_assigned[sex] if
                #                         person.partnership_status == status]
                #     for i, person in enumerate(people_in_status):
                #         print(f'{i + 1}: ID: {person.id}, Age: {person.age}, Partnership Status: {person.partnership_status}, Spouse: {person.spouse}')
                #     print('...............')

            for sex_to_remove, (people_dict, people_to_remove) in {
                'm': [self.ctxt.men_by_age, all_people_assigned['m']],
                'f': [self.ctxt.women_by_age, all_people_assigned['f']]
            }.items():
                self._remove_from_people_dict(people_dict, people_to_remove)

            return all_individuals

        def _find_and_set_partner(self, age_band, sex):
            low_age, high_age = age_band.split('-')
            low_age, high_age = int(low_age), int(high_age)

            num_to_assign = self.ctxt.partnerships_oa.loc[(sex, 'partnered'), age_band]
            if num_to_assign == 0: return
            # print(f"Assigning: {num_to_assign} partners")

            eligible_women = []
            probabilities = []
            for age in filter(lambda x: low_age <= x, self.ctxt.women_by_age.keys()):
                prob = self.ctxt.partnerships_lad.loc[(sex, 'partnered'), age]
                women_at_age = self.ctxt.women_by_age[age]
                eligible_women.extend(women_at_age)
                probabilities.extend([prob] * len(women_at_age))
            if not eligible_women:
                raise Exception("NO ELIGIBLE WOMEN IN AGE RANGE")

            probabilities = probabilities / sum(probabilities)
            chosen_women_indices = np.random.choice(range(len(eligible_women)), size=num_to_assign, replace=False,
                                                    p=probabilities)
            chosen_women = [eligible_women[i] for i in chosen_women_indices]

            # print("___________")
            # print("Chosen women:")
            successfully_matched = {'f': [], 'm': []}
            couples = []
            for i, person in enumerate(chosen_women):
                self._get_hetero_partner(person) if self._get_sexuality(person) == 1 else self._get_same_sex_partner()
                spouse = person.spouse
                # print(f'{i+1}: ID: {person.id}, Sex: {person.sex}, Age: {person.age}, Partnership Status: {person.partnership_status}, Spouse ID: {person.spouse.id}')
                # print(f'Spouse: ID: {spouse.id}, Sex: {spouse.sex}, Age: {spouse.age}, Partnership Status: {spouse.partnership_status}, Spouse ID: {spouse.spouse.id}')
                successfully_matched[person.sex].append(person)
                successfully_matched[spouse.sex].append(spouse)
                couples.append([person, spouse])

            for sex_to_remove, (people_dict, people_to_remove) in {
                'm': [self.ctxt.men_by_age, successfully_matched['m']],
                'f': [self.ctxt.women_by_age, successfully_matched['f']]
            }.items():
                self._remove_from_people_dict(people_dict, people_to_remove)

            return couples

        def _get_random_person_in_age_bracket(self):
            pass

        def _get_sexuality(self, person):
            # TODO: Add logic for determining sexuality... Placeholder is all Heterosexual (1)
            return 1

        def _get_hetero_partner(self, person): # ToDo For cleanliness, I suggest we integrate same sex here too.
            age, sex = person.age, person. sex
            prob_df = self.distributor.couples_age_disparity['hetero']
            prob_age = prob_df[age]
            prob_age = prob_age[prob_age != 0]
            valid_ages = prob_age.index

            spouse_options = self.ctxt.men_by_age if sex == 'f' else self.ctxt.women_by_age

            eligible_spouses = []
            spouse_probabilities = []
            for i, age in enumerate(filter(lambda x: x in valid_ages, self.ctxt.men_by_age.keys())):
                eligible_spouses_at_age = spouse_options[age]
                eligible_spouses.extend(eligible_spouses_at_age)
                spouse_probabilities.extend([prob_age.loc[age]] * len(eligible_spouses_at_age))
            if not eligible_spouses:
                print("NO ELIGIBLE SPOUSES IN AGE RANGE")
                exit()

            spouse_probabilities = spouse_probabilities/sum(spouse_probabilities)
            chosen_spouse = np.random.choice(eligible_spouses, size=1, replace=False, p=spouse_probabilities)[0]
            person.spouse, person.partnership_status = chosen_spouse, 'partnered'
            chosen_spouse.spouse, chosen_spouse.partnership_status = person, 'partnered'

            return chosen_spouse

        def _get_same_sex_partner(self):
            spouse = None # ToDo
            return spouse

    class Kids:
        def __init__(self, people_instance):
            self.people_instance = people_instance

        def _get_sibling(self):
            pass

    class PeopleChecker:
        def __init__(self, people_instance):
            self.people_instance = people_instance

        def _check_age_dict_not_empty(self):
            pass

        def _check_oldpeople_left(self):
            pass

class OldPeopleHousehold(HouseholdDistributor):
    def __init__(self, distributor: HouseholdDistributor):
        self.distributor = distributor
        self.ppl = distributor.ppl

    def create_households(self, sub_collections, ctxt):
        print(f"Size Households: {ctxt.size_households}")
        print(f"Number Households: {ctxt.number_households}")
        print(f"Partnerships by OA: {ctxt.partnerships_oa}")
        print(f"Partnerships by LAD: {ctxt.partnerships_lad}")

        partnership_status_opt = {
            'partnered': ['partnered'],
            'single': ['never_partnered', 'divorced_separated', 'widowed']
        }
        for sub_collection, number_households in zip(sub_collections, ctxt.number_households[sub_collections]):
            print(f"Number of {sub_collection} = {number_households}")
            size = self.distributor.household_comp_collections['old'][sub_collection]['size']
            partnership_status = 'partnered' if size == 2 else 'single'

            potential_households = self.ppl.partnership._status_for_age_bracket(age_band='65-99', partnership_status=partnership_status)

            new_households, potential_households = self.distributor.select_random_household_occupants(potential_households, number_households)

class NoKidsHousehold(HouseholdDistributor):
    def __init__(self, distributor: HouseholdDistributor):
        self.distributor = distributor

class DependentHousehold(HouseholdDistributor):
    def __init__(self, distributor: HouseholdDistributor):
        self.distributor = distributor

class NonDependentHousehold(HouseholdDistributor):
    def __init__(self, distributor: HouseholdDistributor):
        self.distributor = distributor

class OtherHousehold(HouseholdDistributor):
    def __init__(self, distributor: HouseholdDistributor):
        self.distributor = distributor