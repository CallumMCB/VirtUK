from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict
import bisect

from VirtUK import paths
from VirtUK.demography import Person
from VirtUK.geography import Geography
from VirtUK.utils import random_choice_numba


@dataclass(frozen=True)
class DemographyFPs:
    config_path = paths.configs_path

    demography_dir = f'{paths.data_path}/input/demography/'
    age_dir_21 = (demography_dir + 'age_dist_2021/')
    age_total_fp = (age_dir_21 + '1_year_ages_total_oa.csv')
    age_male_fp = (age_dir_21 + '1_year_ages_male_oa.csv')
    age_female_fp = (age_dir_21 + '1_year_ages_female_oa.csv')

    geography_dir = f'{paths.data_path}/input/geography/'
    area_msoa_region_fp = (geography_dir + 'oa_msoa_region.csv')

    m_comorbidity_fp = None
    f_comorbidity_fp = None


class DemographyError(BaseException):
    pass


class AgeSexGenerator:
    def __init__(
        self,
        total_age_counts: list,
        male_age_counts: list,
        female_age_counts: list,
        max_age=99,
    ):
        """
        - male_age_counts is an array where the index in the array indicates the age,
        and the value indicates the number of counts in that age from 0-90 years.
        - Column 'all_ages' tells us how many people are in that area.
        - female_age_counts is the female equivalent of male_age_counts
        - ethnicity_age_bins are the lower edges of the age bins that ethnicity data is in
        ethnicity_groups are the labels of the ethnicities which we have data for.
        Example:
            age_counts = [1, 2, 3, ..., 180] means 1 person of age 0, 2 people of age 1 and 3 people of age 2 and
            a total of 180 people in the output area.
            ethnicity_groups = ['A','B','C'] - there are three types of ethnicities that we are
                                          assigning here.
            ethnicity_structure = [[0,5,3],[2,3,0],...] in the first age bin, we assign people
                                          ethnicities A:B:C with probability 0:5:3, and so on.
        Given this information we initialize two generators for age and sex, that can be accessed
        through gen = AgeSexGenerator().age() and AgeSexGenerator().sex().

        Parameters
        ----------
        total_age_counts
            A list with the total counts for each age in that area.
        male_age_counts
            A list with the male counts for each age in that area.
        female_age_counts
            A list with the female counts for each age in that area.
        """
        self.n_residents = total_age_counts[-1]

        ages = np.repeat(np.arange(0, len(total_age_counts) - 1, dtype=np.int16), total_age_counts[:-1])
        # Store female and male counts for later use
        self.female_counts = np.array(female_age_counts[:-1], dtype=np.int16)
        self.male_counts = np.array(male_age_counts[:-1], dtype=np.int16)

        self.age_iterator = iter(ages)
        self.max_age = max_age

    def age(self) -> int:
        try:
            return min(next(self.age_iterator), self.max_age)
        except StopIteration:
            raise DemographyError("No more people living here!")

    def sex(self, age: int) -> str:
        """
        Determine the sex of an individual based on their age.

        Parameters
        ----------
        age : int
            The age of the individual.

        Returns
        -------
        str
            'f' for female, 'm' for male.
        """
        if self.female_counts[age] > 0:
            self.female_counts[age] -= 1
            return "f"
        else:
            self.male_counts[age] -= 1
            return "m"

    # def ethnicity(self) -> str:
    #     try:
    #         return next(self.ethnicity_iterator)
    #     except StopIteration:
    #         raise DemographyError("No more people living here!")


class Population:
    def __init__(self, people: Optional[List[Person]] = None):
        """
        A population of people.

        Behaves mostly like a list but also has the name of the area attached.

        Parameters
        ----------
        people
            A list of people generated to match census data for that area
        """
        if people is None:
            self.people_dict = {}
            self.people_ids = set()
            self.people = []
        else:
            self.people_dict = {person.id: person for person in people}
            self.people_ids = set(self.people_dict.keys())
            self.people = people

    def __len__(self):
        return len(self.people)

    def __iter__(self):
        return iter(self.people)

    def __getitem__(self, index):
        return self.people[index]

    def __add__(self, population: "Population") -> "Population":
        self.people.extend(population.people)
        self.people_dict = {**self.people_dict, **population.people_dict}
        self.people_ids = set(self.people_dict.keys())
        return self

    def add(self, person: Person):
        self.people_dict[person.id] = person
        self.people.append(person)
        self.people_ids.add(person.id)

    def remove(self, person: Person):
        del self.people_dict[person.id]
        self.people.remove(person)
        self.people_ids.remove(person.id)

    def extend(self, people):
        for person in people:
            self.add(person)

    def get_from_id(self, person_id: int) -> Person:
        return self.people_dict[id]

    @property
    def members(self):
        return self.people

    @property
    def total_people(self):
        return len(self.members)

    @property
    def infected(self):
        return [person for person in self.people if person.infected]

    @property
    def dead(self):
        return [person for person in self.people if person.dead]

    @property
    def vaccinated(self):
        return [person for person in self.people if person.vaccinated]


class ComorbidityGenerator:
    def __init__(self, comorbidity_data):
        self.male_comorbidities_probabilities = np.array(
            comorbidity_data[0].values.T, dtype=np.float64
        )
        self.female_comorbidities_probabilities = np.array(
            comorbidity_data[1].values.T, dtype=np.float64
        )
        self.ages = np.array(comorbidity_data[0].columns).astype(int)
        self.comorbidities = np.array(comorbidity_data[0].index).astype(str)
        self.comorbidities_idx = np.arange(0, len(self.comorbidities))

    def _get_age_index(self, person):
        column_index = 0
        for idx, i in enumerate(self.ages):
            if person.age <= i:
                break
            else:
                column_index = idx
        if column_index != 0:
            column_index += 1
        return column_index

    def get_comorbidity(self, person):
        age_index = self._get_age_index(person)
        if person.sex == "m":
            comorbidity_idx = random_choice_numba(
                self.comorbidities_idx, self.male_comorbidities_probabilities[age_index]
            )
        else:
            comorbidity_idx = random_choice_numba(
                self.comorbidities_idx,
                self.female_comorbidities_probabilities[age_index],
            )
        return self.comorbidities[comorbidity_idx]


class Demography:
    def __init__(
        self,
        area_names: List[str],
        age_sex_generators: Dict[str, AgeSexGenerator],
        comorbidity_data: Optional[Dict] = None,
    ):
        """
        Tool to generate population for a certain geographical region.

        Parameters
        ----------
        area_names : List[str]
            Names of the geographical areas
        age_sex_generators : Dict[str, AgeSexGenerator]
            A dictionary mapping area identifiers to generators that generate
            age and sex for individuals.
        comorbidity_data : Optional[Dict]
            Data related to comorbidity for the individuals
        """
        self.area_names = area_names
        self.age_sex_generators = age_sex_generators
        self.comorbidity_data = comorbidity_data

    def populate(self, area_name: str, ethnicity: bool = True, comorbidity: bool = True) -> Population:
        """
        Generate a population for a given area.

        Parameters
        ----------
        area_name : str
            The name of an area for which a population should be generated
        ethnicity : bool
            Whether to assign ethnicity information to individuals
        comorbidity : bool
            Whether to assign comorbidity data to individuals

        Returns
        -------
        Population
            A population object containing generated individuals
        """
        age_and_sex_generator = self.age_sex_generators.get(area_name)
        if not age_and_sex_generator:
            raise ValueError(f"Area '{area_name}' not found in age_sex_generators.")

        comorbidity_generator = ComorbidityGenerator(self.comorbidity_data) if comorbidity else None
        generated_people = []

        for _ in range(age_and_sex_generator.n_residents):
            current_age = age_and_sex_generator.age()
            person = Person.from_attributes(
                age=current_age,
                sex=age_and_sex_generator.sex(current_age),
                # ethnicity=age_and_sex_generator.ethnicity() if ethnicity else None,
            )
            if comorbidity and comorbidity_generator:
                person.comorbidity = comorbidity_generator.get_comorbidity(person)
            generated_people.append(person)
        return Population(people=generated_people)

    @classmethod
    def for_geography(
        cls,
        geography: Geography,
        fps: DemographyFPs = DemographyFPs(),
        config: Optional[dict] = None,
    ) -> "Demography":
        """
        Initializes demography from an existing geography.

        Parameters
        ----------
        geography : Geography
            An instance of the Geography class
        fps : DemographyFPs
            File paths for demographic data
        config : Optional[dict]
            Configuration settings (unused currently)

        Returns
        -------
        Demography
            A demography instance for the given geography
        """
        if not geography.areas:
            raise DemographyError("This Geography has no areas!")
        area_names = [area.name for area in geography.areas]
        return cls.for_areas(area_names, fps, config)

    @classmethod
    def for_zone(
            cls,
            filter_key: Dict[str, list],
            fps: DemographyFPs = DemographyFPs(),
            config: Optional[dict] = None,
    ) -> "Demography":
        """
        Initializes a geography for a specific list of zones.

        Parameters
        ----------
        filter_key : Dict[str, list]
            A dictionary containing the type of zone and corresponding zone names
        fps : DemographyFPs
            File paths for demographic data
        config : Optional[dict]
            Configuration settings (unused currently)

        Returns
        -------
        Demography
            A demography instance for the filtered zones
        """
        if len(filter_key.keys()) > 1:
            raise NotImplementedError("Only one type of area filtering is supported.")
        geo_hierarchy = pd.read_csv(fps.area_msoa_region_fp)
        zone_type, zone_list = filter_key.popitem()
        area_names = geo_hierarchy[geo_hierarchy[zone_type].isin(zone_list)]["area"].tolist()
        if not area_names:
            raise DemographyError("Region returned an empty area list.")
        return cls.for_areas(area_names, fps, config)


    @classmethod
    def for_areas(
        cls,
        area_names: List[str],
        fps: DemographyFPs,
        config: Optional[dict] = None,
    ) -> "Demography":
        """
        Load data from files and construct classes capable of generating demographic
        data for individuals in the population.

        Parameters
        ----------
        area_names : List[str]
            List of areas for which to create a demographic generator
        fps : DemographyFPs
            File paths for demographic data
        config : Optional[dict]
            Configuration settings (unused currently)

        Returns
        -------
        Demography
            A demography instance representing the specified areas
        """
        age_sex_generators = cls._load_age_and_sex_generators(fps.age_total_fp, fps.age_male_fp, fps.age_female_fp,  area_names)
        comorbidity_data = cls._load_comorbidity_data(m_comorbidity_path=None, f_comorbidity_path=None)
        return cls(area_names=area_names, age_sex_generators=age_sex_generators, comorbidity_data=comorbidity_data)

    @staticmethod
    def _load_age_and_sex_generators(
            total_age_counts: str,
            male_age_counts: str,
            female_age_counts: str,
            area_names: List[str],
    ) -> Dict[str, AgeSexGenerator]:
        """
        A dictionary mapping area identifiers to a generator of age and sex.

        Parameters
        ----------
        male_age_counts : str
            Path to the age structure data for all people
        female_age_counts : str
            Path to the age structure data for females
        area_names : List[str]
            List of area names for which to generate data

        Returns
        -------
        Dict[str, AgeSexGenerator]
            A dictionary of age and sex generators for each area
        """
        total_age_counts_df = pd.read_csv(total_age_counts, index_col=0)
        total_age_counts_df = total_age_counts_df.loc[area_names]

        male_age_counts_df = pd.read_csv(male_age_counts, index_col=0)
        male_age_counts_df = male_age_counts_df.loc[area_names]

        female_age_counts_df = pd.read_csv(female_age_counts, index_col=0)
        female_age_counts_df = female_age_counts_df.loc[area_names]

        ret = {}
        for ((index, total_age_counts), (_, male_age_counts), (_, female_age_counts)) in zip(
                total_age_counts_df.iterrows(),
                male_age_counts_df.iterrows(),
                female_age_counts_df.iterrows(),
        ):
            ret[index] = AgeSexGenerator(total_age_counts, male_age_counts.values, female_age_counts.values)

        return ret

    @staticmethod
    def _load_comorbidity_data(m_comorbidity_path=None, f_comorbidity_path=None):
        if m_comorbidity_path is not None and f_comorbidity_path is not None:
            male_co = pd.read_csv(m_comorbidity_path)
            female_co = pd.read_csv(f_comorbidity_path)

            male_co = male_co.set_index("comorbidity")
            female_co = female_co.set_index("comorbidity")

            for column in male_co.columns:
                m_nc = male_co[column].loc["no_condition"]
                m_norm_1 = 1 - m_nc
                m_norm_2 = np.sum(male_co[column]) - m_nc

                f_nc = female_co[column].loc["no_condition"]
                f_norm_1 = 1 - f_nc
                f_norm_2 = np.sum(female_co[column]) - f_nc

                for idx in list(male_co.index)[:-1]:
                    male_co[column].loc[idx] = (
                            male_co[column].loc[idx] / m_norm_2 * m_norm_1
                    )
                    female_co[column].loc[idx] = (
                            female_co[column].loc[idx] / f_norm_2 * f_norm_1
                    )

            return [male_co, female_co]

        else:
            return None