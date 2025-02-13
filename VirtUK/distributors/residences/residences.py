import numpy as np
import pandas as pd
import random

from tabulate import tabulate
from pprint import pprint
from tqdm import tqdm
import logging

from typing import Dict, List, Tuple, Any, Optional

from VirtUK.demography import Person
from VirtUK.geography import Area, LAD
from VirtUK.groups import Household, Households
from VirtUK import DataLoader, AgeLimits
from . import HouseholdContext, People_Assignment

test_specific = ["E00171869"]
logger = logging.getLogger("household_distributor")

class ResidenceDistributor:
    def __init__(
            self,
            age_counts: Dict[str, pd.DataFrame],

            hierarchy: pd.DataFrame,

            household_comp_config: Dict[str, Any],
            household_comp_numbers: pd.DataFrame,
            household_comp_collections: Dict[str, pd.DataFrame],
            household_sizes: pd.DataFrame,

            oa_communal_residents: pd.DataFrame,
            oa_students: pd.DataFrame,
            msoa_students: pd.DataFrame,
            msoa_communal_residents: pd.DataFrame,
            msoa_communal_types: pd.DataFrame,
            msoa_staff_temporary_residents: pd.DataFrame,
            cqc_care_homes: pd.DataFrame,
            prisons: pd.DataFrame,

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
        self.oa_students = oa_students
        self.msoa_students = msoa_students
        self.msoa_communal_residents = msoa_communal_residents
        self.msoa_communal_types = msoa_communal_types
        self.msoa_staff_temporary_residents = msoa_staff_temporary_residents

        self.cqc_care_homes = cqc_care_homes
        self.prisons = prisons

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
    )-> "ResidenceDistributor":
        all_data = data_loader.load_all_data()
        return cls(
            age_counts = all_data['ac'],
            hierarchy = all_data['hrcy'],
            household_comp_config = all_data['hh_comps']['config'],
            household_comp_numbers = all_data['hh_comps']['numbers'],
            household_comp_collections = all_data['hh_comps']['collections'],
            oa_communal_residents = all_data['oa_cd'],
            oa_students = all_data['oa_stdnts'],
            msoa_students = all_data['msoa_stdnts'],
            msoa_communal_residents = all_data['msoa_cd']['residents_by_sex'],
            msoa_communal_types = all_data['msoa_cd']['communal_type'],
            msoa_staff_temporary_residents = all_data['msoa_cd']['staff_or_temporary_residents'],
            cqc_care_homes = all_data['cqc_chs'],
            prisons = all_data['prisons'],
            household_sizes = all_data['hh_sizes'],
            child_parent_age = all_data['msoa_cpad'],
            couples_age_disparity = all_data['cads'],
            oa_partner_status= all_data['oa_ptnr_status'],
            lad_partner_status = all_data['lad_ptnr_status'],
            lad_dependent_children = all_data['lad_dc'],
            age_limits = age_limits
        )

    def distribute_people_and_residences_to_areas(
            self,
            lads: List[LAD]
    ):
        logger.info("Distributing people to households")

        area_names = [area.name for lad in lads for area in lad.areas]
        msoa_names = [msoa.name for lad in lads for msoa in lad.super_areas]
        print(msoa_names)

        # Get data for all areas using the list of area names
        household_numbers_df = self.household_comp_numbers.loc[area_names]
        household_sizes_df = self.household_sizes.loc[area_names]
        student_accommodation = self.oa_students.loc[area_names]
        oa_partnership_status_df = self.oa_partner_status.loc[area_names]
        broad_ages = self.age_counts['broad'].loc[area_names]

        for lad in lads:
            partnerships_lad = self.lad_partner_status.loc[lad.name]
            dependent_kids_lad = self.lad_dependent_children.loc[lad.name]
            for counter, msoa in enumerate(
                tqdm(
                    lad.super_areas,
                    total=len(lad.areas),
                    desc=f"Distributing in {lad.name}"),
                    start=1
                ):


                print(f"###########   MSOA: {msoa.name}   ###########")
                if msoa.name in self.msoa_communal_residents.index:
                    communal_df = self.msoa_communal_residents.loc[msoa.name]
                    communal_df['Total'] = communal_df.sum(numeric_only=True, axis=1)
                    total_row = communal_df.sum(numeric_only=True, axis=0)
                    total_row.name = 'Total'
                    print(tabulate(communal_df, headers="keys", tablefmt="psql"))
                if msoa.name in self.msoa_communal_types.index:
                    df = self.msoa_communal_types.loc[msoa.name]
                    filtered_df = df.loc[:, (df != 0).any()]
                    print(tabulate(filtered_df, headers="keys", tablefmt="psql"))

                if msoa.name in self.cqc_care_homes.index:
                    ch_cqc = self.cqc_care_homes.loc[msoa.name]
                    if not ch_cqc.empty:
                        if isinstance(ch_cqc, pd.Series):
                            ch_cqc = ch_cqc.to_frame().T
                        print(tabulate(ch_cqc, headers="keys", tablefmt="psql"))
                if msoa.name in self.prisons.index:
                    prisons = self.prisons.loc[msoa.name]
                    if not prisons.empty:
                        if isinstance(prisons, pd.Series):
                            prisons = prisons.to_frame().T
                        print(tabulate(prisons, headers="keys", tablefmt="psql"))

                # Iterate through each area within the current MSOA
                for area in msoa.areas:
                    # If a test-specific filter is applied, skip areas not in that list.
                    if test_specific and area.name not in test_specific:
                        continue

                    ctxt = HouseholdContext()
                    ctxt.area = area
                    ctxt.all_households = []
                    # Use direct lookup with the area name
                    ctxt.number_households = household_numbers_df.loc[area.name]
                    ctxt.size_households = household_sizes_df.loc[area.name].to_numpy()
                    ctxt.student_accommodation = student_accommodation.loc[area.name]
                    ctxt.partnerships_oa = oa_partnership_status_df.loc[area.name]
                    ctxt.partnerships_lad = partnerships_lad
                    ctxt.dependent_kids_lad = dependent_kids_lad
                    ctxt.broad_ages = broad_ages.loc[area.name]

                    self.ppl = People_Assignment(self, ctxt)
                    area.households = self.distribute_people_to_households(ctxt)

    def distribute_people_to_households(self, ctxt)-> Households:
        collection_distributors = {
            'old': OldPeopleHousehold,
            # 'nokids': NoKidsHousehold,
            # 'd_family': DependentHousehold,
            # 'nd_family': NonDependentHousehold,
            # 'other': OtherHousehold,
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

class OldPeopleHousehold(ResidenceDistributor):
    def __init__(self, distributor: ResidenceDistributor):
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