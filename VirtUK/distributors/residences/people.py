from ._household_context import HouseholdContext
from .temporary_households import Temp_OA_Households
from VirtUK.geography import Area

from tabulate import tabulate
from pprint import pprint

import pandas as pd
from collections import defaultdict
import numpy as np


class People_Assignment:
    def __init__(self, household_distributor, context: HouseholdContext):
        """
        Initialize People instance with given context and distributor.

        Parameters
        ----------
        context : HouseholdContext
            The context containing data and configurations for household distributions.
        """
        self.ctxt = context
        self.distributor = household_distributor

        self._create_people_dicts(self.ctxt.area)

        # Nested helper classes
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
            """
            Perform initial assignment of people to households and summarize population.

            Parameters
            ----------
            people_instance : People
                The parent People instance.
            """
            self.ppl_inst = people_instance
            self.ctxt = self.ppl_inst.ctxt

            # Print area and initial population breakdown
            print(f"\n___ Output Area {self.ctxt.area.name} ___\n")
            print("\n=== Population Breakdown ===")
            # Summarize population and print the result
            self.smrzd_pop, self.transition_population, print_statement = self.summarize_population()
            print(print_statement)

            # Print household size and available households
            self._print_household_info()

            # Setup base configuration (fixed households, etc.)
            self.base_configuration()

        def summarize_population(self):
            # Summarize population by predefined age groups
            young_student_ages = ['16-17', '18-20', '21-24']
            adult_student_ages = ['25-29', '30+']

            y_student_communal = self.ctxt.student_accommodation.loc[
                ['communal establishment: Other', 'communal establishment: University'], young_student_ages].sum().sum()
            a_student_communal = self.ctxt.student_accommodation.loc[
                ['communal establishment: Other', 'communal establishment: University'], adult_student_ages].sum().sum()
            y_student_only = self.ctxt.student_accommodation.loc[['all student household'], young_student_ages].sum().sum()
            a_student_only = self.ctxt.student_accommodation.loc[['all student household'], adult_student_ages].sum().sum()
            y_student_single = self.ctxt.student_accommodation.loc[['Living alone'], young_student_ages].sum().sum()
            y_student_other = self.ctxt.student_accommodation.loc[['Living in another household type'], young_student_ages].sum().sum()
            y_student_parents = self.ctxt.student_accommodation.loc[['parents'], young_student_ages].sum().sum()

            summary_counts = {
                "Kid": self.ctxt.broad_ages.loc["0-4"] + self.ctxt.broad_ages.loc["5-9"] + self.ctxt.broad_ages.loc["10-15"],
                "Youth": self.ctxt.broad_ages.loc["16-19"] + self.ctxt.broad_ages.loc["20-24"] - y_student_communal - y_student_only - y_student_single - y_student_other - y_student_parents,
                "Students": y_student_only + a_student_only + y_student_single + y_student_other,
                "Adult":  self.ctxt.broad_ages.loc["25-34"] + self.ctxt.broad_ages.loc["35-49"] + self.ctxt.broad_ages.loc["50-64"] - a_student_communal - a_student_only,
                "Old": self.ctxt.broad_ages.loc["65-74"] + self.ctxt.broad_ages.loc["75-84"] + self.ctxt.broad_ages.loc["85-99"]
            }

            # Create an array representation
            population_array = np.array([
                summary_counts["Kid"],
                summary_counts["Youth"],
                summary_counts["Students"],
                summary_counts["Adult"],
                summary_counts["Old"]
            ])
            print(tabulate(self.ctxt.student_accommodation,  headers='keys', tablefmt='pipe'))

            transition_population = summary_counts['Youth']

            # Generate summary output
            summary_output = "\nFrom Broad:\n"
            for category, count in summary_counts.items():
                summary_output += f"  {category}: {count}\n"

            return population_array, transition_population, summary_output

        def base_configuration(self):
            print("\n=== Assignment of Fixed Households ===")
            print("\nAvailable People:")
            print(pd.DataFrame(np.reshape(self.smrzd_pop, (-1, 5)), columns = ['Kids', 'YA', 'students', 'Adults', 'Old']).to_string(index=False))
            fixed_hhs = {
                key: values
                for key, values in self.ppl_inst.distributor.config_size_households['fixed'].items()
                if key in self.available_hhs.index
            }
            single_students = self.ctxt.student_accommodation.loc[['Living alone'], ['16-17', '18-20', '21-24']].sum().sum()
            if single_students > 0:
                self.available_hhs['0 0 1 0'] -= single_students
                fixed_hhs['0 1 0 0'] = {'size': 1, 'comp': np.array([0, 0, 1, 0, 0]), 'type': 'Student'}

            overall_target = np.array([0] + list(self.size_hhs[self.size_hhs.index != 'Total']['Number of Households']))

            print("\nFixed Households:")
            pprint(fixed_hhs)
            for key, values in fixed_hhs.items():
                if key == '0 1 0 0':
                    number_hhs = single_students
                else:
                    number_hhs = self.available_hhs[key]
                self.smrzd_pop -= number_hhs * values['comp']
                self.size_hhs.loc[values['size']] -= number_hhs

            print("\nRemaining Available People:")
            print(pd.DataFrame(np.reshape(self.smrzd_pop, (-1, 5)), columns = ['Kids', 'YA', 'students', 'Adults', 'Old']).to_string(index=False))
            print("\nRemaining Sizes of Households:")
            print(self.size_hhs.T)

            print("\n##### Running Temp OA Households ######\n")
            fixed_comp_number = {
                key: {'comp': values['comp'], 'size': values['size'], 'type': values['type'], 'number': self.available_hhs[key]}
                for key, values in self.ppl_inst.distributor.config_size_households['fixed'].items()
                if key in self.available_hhs.index
            }
            if single_students > 0:
                fixed_comp_number['0 1 0 0'] = {'comp': np.array([0, 0, 1, 0, 0]), 'size': 1, 'type': 'Student', 'number': single_students}

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
                transition_population=self.transition_population,
                var_target_dist=var_target,
                students = self.ctxt.student_accommodation,
                dependent_kids_ratios=self.ctxt.dependent_kids_lad,
                num_simple_family_hhs=100
            )
            base_temp_oa.initialize()

            print("\n--- Student Accommodation ---\n")
            print(tabulate(base_temp_oa.students, headers='keys', tablefmt = 'pipe'))
            print("\n--- Fixed Households ---\n")
            base_temp_oa._print_grouped_households_table(base_temp_oa.fixed_households)
            # Specify the number of branches you want
            num_branches = 3
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


        def _print_household_info(self):
            """Print information about household sizes and availability."""
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
            print(tabulate(pd.DataFrame(self.available_hhs), headers='keys', tablefmt='pipe'))



