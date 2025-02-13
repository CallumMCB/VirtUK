import numpy as np
import random
import pandas as pd
from tabulate import tabulate
import copy

from . import Chi2Analysis, MetropolisHastingsAnalysis, Variable_Household, Fixed_Household

type_map = {0: "kid", 1: "youth", 2: "student", 3: "adult", 4: "old"}
rvrs_map = {"kid": 0, "youth": 1, "student": 2, "adult": 3, "old": 4}

class Temp_OA_Households:
    def __init__(
            self,
            area,
            overall_target_dist,
            fixed_hh_composition: dict,
            var_hh_composition: dict,
            person_counts: np.ndarray,
            transition_population: int,
            var_target_dist: np.ndarray,
            students: pd.DataFrame,
            dependent_kids_ratios: pd.DataFrame,
            num_simple_family_hhs):
        # Initialize instance attributes that will stay the same across runs
        self.area = area
        self.overall_target_dist = overall_target_dist
        self.largest_size = next((i for i in range(len(self.overall_target_dist) - 1, -1, -1) if self.overall_target_dist[i] != 0), None)
        self.fixed_household_composition = fixed_hh_composition
        self.variable_household_composition = var_hh_composition
        self.person_counts = person_counts
        self.transition_population = transition_population
        self.transitioned = 0
        self.var_target_dist = var_target_dist
        self.students = students
        self.dependent_kids_ratios = dependent_kids_ratios
        self.num_simple_family_hhs = num_simple_family_hhs

        self.students_target = {
            "all students": students.loc[["all student household"]].sum().sum(),
            "parents": students.loc[["parents"], ['16-17','18-20','21-24']].sum().sum(),
            "other": students.loc[["Living in another household type"], ['16-17','18-20','21-24']].sum().sum(),
        }

        # Attributes that are initialized once and retained across multiple calls to run()
        self.fixed_households = []
        self.fixed_distribution = np.zeros(9)
        self.households = []
        index_values = [str(v['comp']) for v in var_hh_composition.values()]
        self.composition_totals = pd.DataFrame(0, index=index_values, columns=['kid', 'youth', 'student', 'adult', 'old'], dtype=np.int16)
        self.accepting_households = {ptype: set() for ptype in type_map.values()}
        self.donating_households = {ptype: set() for ptype in type_map.values()}
        self.household_counter = 0
        self.intrinsic_error = 0
        self.chi2_result = None

        # Set up Metropolis-Hastings and Chi2Analysis only once per instance
        self.MetHast = None
        self.Analysis = None

    def initialize(self):
        """Method to initialize the fixed households and other components that only need to be set up once."""
        self._generate_fixed_households(self.fixed_household_composition, self.overall_target_dist)
        self._initialize_households(self.variable_household_composition)
        self._assign_initial_people(self.person_counts)

        # Set up Metropolis-Hastings and Chi2Analysis only once per instance
        self.MetHast = MetropolisHastingsAnalysis(self)
        self.Analysis = Chi2Analysis.from_statistics(self,
                                          self.dependent_kids_ratios,
                                          self.variable_household_composition)

    def run(self):
        # Randomly distribute remaining people
        self._distribute_remaining_people(self.person_counts)
        # Run Metropolis-Hastings algorithm after distributing remaining people
        self.MetHast.run(iterations = 4 * len(self.households), initial_temperature=1, cooling_rate=0.995)
        # Print Final Results
        print("\n+++ Variable Household Size Summary +++\n")
        self._print_household_size_summary(distribution=self.variable_distribution, target_dist=self.var_target_dist)
        print("\n+++ Overall Household Size Summary +++\n")
        self._print_household_size_summary(distribution=(self.variable_distribution + self.fixed_distribution), target_dist=self.overall_target)
        print("\n--- Variable Households ---\n")
        self._print_grouped_households_table(self.households)

        self.Analysis.analyze_kids_distribution()

    def branch(self):
        """Create a new instance that is a deep copy of this instance."""
        return copy.deepcopy(self)

    def _generate_fixed_households(self, household_fixed_composition, overall_target_dist):
        """Initialize households based on compositions (only once)."""
        for key, values in household_fixed_composition.items():
            comp, size, number = values['comp'], values['size'], values['number']
            for _ in range(number):
                self.household_counter += 1
                household = Fixed_Household(self.area, self.household_counter, comp, size)
                self.fixed_households.append(household)
        self.overall_target = overall_target_dist
        self.fixed_distribution = self._calculate_initial_distribution(self.fixed_households)

    def _initialize_households(self, household_composition):
        """Initialize households based on compositions."""
        for key, values in household_composition.items():
            comp, min_size, number = values['comp'], values['size'], values['number']

            for _ in range(number):
                self.household_counter += 1
                household = Variable_Household(self, comp, min_size)
                self.households.append(household)
                self._update_household_status(household)

    def _assign_initial_people(self, person_counts):
        """Assign fixed and minimum people counts to households."""
        for household in self.households:
            required = household.fill_minimum_requirement()
            for person_type, required_count in required.items():
                person_type_id = rvrs_map[person_type]
                # if person_type_id == 1 and (person_counts[1] + person_counts[2]) >= required_count:
                #     students = person_counts[2]
                #     if students >= required_count:
                #         person_counts[2] -= required_count
                #         household.add_person('student', required_count)
                #     else:
                #         person_counts[1] -= (required_count - students)
                #         household.add_person('student', students)
                #         household.add_person('youth',  (required_count - students))
                #         person_counts[2] = 0
                if person_counts[person_type_id] >= required_count:
                    household.add_person(person_type, required_count)
                    person_counts[person_type_id] -= required_count
                    self._update_household_status(household)
                elif person_type_id == 0 and person_counts[1] >= required_count:
                    household.add_person(person_type, required_count)
                    person_counts[1] -= required_count
                elif person_type_id == 3 and person_counts[3] + person_counts[4] >= required_count:
                    household.add_person(person_type, required_count)
                    person_counts[4] -= (required_count - person_counts[3])
                    person_counts[3] = 0
                elif person_type_id == 3 and person_counts[3] + self.transition_population >= required_count:
                    household.add_person(person_type, required_count)

                    transitioned = required_count - person_counts[3]
                    person_counts[1] -= transitioned
                    person_counts[3] = 0
                    self.transition_population -= transitioned

                else:
                    print(person_counts[1], person_counts[2], person_counts[3])
                    raise ValueError(f"Not enough {person_type} to satisfy requirements. Missing {required_count - person_counts[person_type_id]}.")

    def _distribute_remaining_people(self, person_counts):
        """Randomly distribute remaining people to eligible households."""
        for household in self.households:
            requirements, _ = household.check_requirements()
            if 'min_required_people' in requirements:
                for _ in range(requirements['min_required_people']):
                    for person_type_id, count in enumerate(person_counts):
                        person_type = type_map[person_type_id]
                        if count > 0 and household.can_accept(person_type):
                            household.add_person(person_type)
                            person_counts[person_type_id] -= 1
                            self._update_household_status(household)
                            break
                    else:
                        raise ValueError(f"Cannot Fulfill Requirements for household {household.composition}, id {household.id}")

        for category, count in enumerate(person_counts):
            while count > 0:
                accepting_households = list(self.accepting_households[type_map[category]])
                if not accepting_households:
                    print(f"ERROR: No accepting households for {count} {type_map[category]}s")
                    count = 0
                    exit()
                else:
                    chosen_household = random.choice(accepting_households)
                    chosen_household.add_person(type_map[category])
                    count -= 1
                    self._update_household_status(chosen_household)

        for household in self.households:
            household.check_requirements()
            self.intrinsic_error += household.intrinsic_error
            self.composition_totals.loc[str(household.composition)] += list(household.num_people.values())

        self.variable_distribution = self._calculate_initial_distribution(self.households)
        print("\n+++ Starting Variable Household Size Summary +++\n")
        self._print_household_size_summary(distribution=self.variable_distribution, target_dist=self.var_target_dist)
        print("\n+++ Starting Household Size Summary +++\n")
        self._print_household_size_summary(distribution=(self.variable_distribution + self.fixed_distribution), target_dist=self.overall_target)
        print("\n--- Starting Variable Households ---\n")
        self._print_grouped_households_table(self.households)
        print("\n--- Starting Variable Household Totals ---\n")
        print(tabulate(self.composition_totals, headers='keys', tablefmt='pipe'))

    def _update_household_status(self, household):
        """Update the status of a household for accepting or donating individuals."""
        for person_type in type_map.values():
            if household.can_accept(person_type):
                self.accepting_households[person_type].add(household)
            else:
                self.accepting_households[person_type].discard(household)

            if household.can_donate(person_type):
                self.donating_households[person_type].add(household)
            else:
                self.donating_households[person_type].discard(household)

    def _donate_person_with_transition(self, donor, target, original_person_type, target_person_type, reverse=False):
        """Donate a person from one household to another with possible transition."""
        intrinsic_error_change = 0

        # Check transition limits
        if original_person_type != target_person_type:
            if not reverse:
                if original_person_type == 'adult' and target_person_type == 'youth':
                    if self.transitioned <= 0:
                        return None  # Cannot perform transition, limit reached
                elif original_person_type == 'youth' and target_person_type == 'adult':
                    if self.transitioned >= self.transition_population:
                        return None  # Cannot perform transition back, no transitions occurred
            else:
                # When reversing, ensure we don't reverse beyond limits
                if original_person_type == 'adult' and target_person_type == 'youth':
                    if self.transitioned >= self.transition_population:
                        return None  # Cannot reverse, no transitions occurred
                elif original_person_type == 'youth' and target_person_type == 'adult':
                    if self.transitioned <= 0:
                        return None  # Cannot reverse beyond transition limit

        if not reverse:
            # Remove person from donor
            intrinsic_error_change += donor.remove_person(original_person_type)
            self.composition_totals.loc[str(donor.composition), original_person_type] -= 1
            # Add person to target
            intrinsic_error_change += target.add_person(target_person_type)
            self.composition_totals.loc[str(target.composition), target_person_type] += 1
        else:
            # Add person back to donor
            intrinsic_error_change += donor.add_person(original_person_type)
            self.composition_totals.loc[str(donor.composition), original_person_type] += 1
            # Remove person from target
            intrinsic_error_change += target.remove_person(target_person_type)
            self.composition_totals.loc[str(target.composition), target_person_type] -= 1

        # Handle transitions
        if original_person_type != target_person_type:
            if original_person_type == 'adult' and target_person_type == 'youth':
                if not reverse:
                    self.transitioned -= 1
                    # print(f"Transitioned adult back to youth. Total transitions: {self.transitioned}")
                else:
                    self.transitioned += 1
                    # print(f"Reversed transition from adult back to ya. Total transitions: {self.transitioned}")
            elif original_person_type == 'youth' and target_person_type == 'adult':
                if not reverse:
                    self.transitioned += 1
                    # print(f"Transitioned a youth to adult. Total transitions: {self.transitioned}")
                else:
                    self.transitioned -= 1
                    # print(f"Reversed a transition from youth to adult. Total transitions: {self.transitioned}")

        return intrinsic_error_change

    @staticmethod
    def _calculate_initial_distribution(households):
        """Calculate the initial distribution of household sizes."""
        size_counts = np.zeros(9).astype(np.int16)
        for household in households:
            size = min(8, household.total_size)
            size_counts[size] += 1
        return size_counts

    @staticmethod
    def _print_grouped_households_table(households):
        """Print households grouped by composition type in a table format."""
        grouped_households = {}
        for household in households:
            comp_key = tuple(household.composition)
            grouped_households.setdefault(comp_key, []).append(household)

        compositions = list(grouped_households.keys())
        composition_labels = [f"Composition {comp}" for comp in compositions]
        max_households = max(len(households) for households in grouped_households.values())

        data = []
        for i in range(max_households):
            row = [repr(grouped_households[comp][i]) if i < len(grouped_households[comp]) else " " for comp in compositions]
            data.append(row)

        df = pd.DataFrame(data, columns=composition_labels)
        print(tabulate(df, headers='keys', tablefmt='pipe'))

    @staticmethod
    def _print_household_size_summary(distribution, target_dist):
        """Print summary of how many households of each size have been produced."""
        difference = distribution - target_dist

        data = {
            f'{i}': [distribution[i], target_dist[i], difference[i]]
            for i in range(len(distribution))
        }
        index = ['current', 'target', 'difference']

        df = pd.DataFrame(data, index=index)

        print(tabulate(df, headers='keys', tablefmt='pipe'))
        print()

    def __repr__(self):
        households_str = "\n".join(repr(household) for household in self.households)
        return (
            f"Temporary_OA_Households(\n"
            f"total_households={len(self.households)},\n"
            f"Households:\n{households_str}\n"
            f")"
        )













