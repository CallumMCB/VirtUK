import numpy as np
import random
import pandas as pd
from tabulate import tabulate
import copy
from pprint import pprint

# Adding the type mapping dictionary
type_map = {0: "kid", 1: "youth", 2: "adult", 3: "old"}
rvrs_map = {"kid": 0, "youth": 1, "adult": 2, "old": 3}


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

        # Attributes that are initialized once and retained across multiple calls to run()
        self.fixed_households = []
        self.fixed_distribution = np.zeros(9)
        self.households = []
        index_values = [str(v['comp']) for v in var_hh_composition.values()]
        self.composition_totals = pd.DataFrame(0, index=index_values, columns=['kids', 'youth', 'adult', 'old'], dtype=np.int16)
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
        self.Analysis = Chi2Analysis.init(self,
                                          self.dependent_kids_ratios,
                                          self.variable_household_composition)

    def run(self):
        # Randomly distribute remaining people
        self._distribute_remaining_people(self.person_counts)
        # Run Metropolis-Hastings algorithm after distributing remaining people
        self.MetHast.run(iterations=4 * len(self.households), initial_temperature=1, cooling_rate=0.995)
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
                household = Temp_Household(self, comp, min_size)
                self.households.append(household)
                self._update_household_status(household)


    def _assign_initial_people(self, person_counts):
        """Assign fixed and minimum people counts to households."""
        for household in self.households:
            required = household.fill_minimum_requirement()
            for person_type, required_count in required.items():
                person_type_id = rvrs_map[person_type]
                if person_counts[person_type_id] >= required_count:
                    household.add_person(person_type, required_count)
                    person_counts[person_type_id] -= required_count
                    self._update_household_status(household)
                elif person_type_id == 1 and person_counts[1] + self.transition_population >= required_count:
                    household.add_person(person_type, required_count)

                    transitioned = required_count - person_counts[1]
                    person_counts[2] -= transitioned
                    person_counts[1] = 0
                    self.transition_population -= transitioned

                else:
                    raise ValueError(f"Not enough {person_type} to satisfy requirements")

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
                        raise ValueError(f"Cannot Fulfill Requirements for household {household.id}")

        for category, count in enumerate(person_counts):
            while count > 0:
                accepting_households = list(self.accepting_households[type_map[category]])
                if not accepting_households:
                    raise ValueError(f"No accepting households for {type_map[category]}")
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
        error_change = 0

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
            error_change += donor.remove_person(original_person_type)
            # Add person to target
            error_change += target.add_person(target_person_type)
        else:
            # Remove person from donor
            error_change += donor.add_person(original_person_type)
            # Add person to target
            error_change += target.remove_person(target_person_type)

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

        return error_change

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


class Fixed_Household(Temp_OA_Households):
    __slots__ = (
        "area",
        "id",
        "composition",
        "num_people",
        "total_size",
    )

    def __init__(self, area, id, composition, size):
        self.area = area
        self.id = id
        self.composition = composition
        self.total_size = size

        self.num_people = {category: composition[i] for i, category in type_map.items()}

    def __repr__(self):
        num_people = [self.num_people[category] for category in type_map.values()]
        return f"id:{self.id}, total_size:{self.total_size}, People: {num_people}"


class Temp_Household(Temp_OA_Households):
    __slots__ = (
        "area",
        "id",
        "allowed_types",
        "composition",
        "num_people",
        "total_size",
        "fixed_types",
        "variable_types",
        "min_size",
        "conditions_satisfied",
        "intrinsic_error",
        "donation_weight",
        "accepting_weight"
    )

    def __init__(self, parent, composition, min_size):
        self.area = parent.area
        self.id = parent.household_counter
        self.composition = composition
        self.min_size = min_size
        self.total_size = 0
        self.conditions_satisfied = False
        self.intrinsic_error = 0
        self.donation_weight = 1
        self.accepting_weight = 1

        # Mapping of categories
        self.num_people = {category: 0 for category in type_map.values()}

        # Parse composition to set allowed types
        self._parse_types(composition)

        self.checks = self.Checks(self)
        self.TOAH = parent

    def _parse_types(self, composition):
        """Parse the composition to set allowed types."""
        ft_idxs = np.where((composition != 0) & (composition % 1 == 0))[0]
        vt_idxs = np.where((composition % 1 != 0))[0]
        self.fixed_types = {type_map[i]: composition[i] for i in ft_idxs}
        self.variable_types = {type_map[i]: int(np.floor(composition[i])) for i in vt_idxs}

    def fill_minimum_requirement(self):
        """Calculate the minimum number of people needed to satisfy requirements."""
        required = {}
        for person_type, minimum in {**self.fixed_types, **self.variable_types}.items():
            if self.num_people[person_type] < minimum:
                required[person_type] = int(minimum - self.num_people[person_type])
        return required

    def can_accept(self, person_type):
        """Check if a household can accept a person of the given type."""
        return person_type in self.variable_types

    def can_donate(self, person_type):
        """Check if the household can donate a person of the given type."""
        return person_type in self.variable_types and self.num_people[person_type] > self.variable_types[person_type]

    def add_person(self, person_type, number=1):
        """Add a person to the household."""
        self.num_people[person_type] += number
        self.total_size += number
        _, error_change = self.check_requirements()
        return error_change

    def remove_person(self, person_type):
        """Remove a person from the household."""
        self.num_people[person_type] -= 1
        self.total_size -= 1
        _, error_change = self.check_requirements()
        return error_change

    def reset_weights_and_errors(self):
        """Reset the weights and errors to their initial values."""
        self.intrinsic_error = 0
        self.donation_weight = 1
        self.accepting_weight = 1

    def check_requirements(self):
        """Check if the household meets all requirements and update weights and errors."""
        original_intrinsic_error = self.intrinsic_error
        self.reset_weights_and_errors()

        requirements = {}

        # Perform checks using the embedded Checks class
        self.checks.check_min_size(requirements)
        self.checks.perform_checks()

        # Update conditions satisfied
        self.conditions_satisfied = not requirements
        return requirements, self.intrinsic_error - original_intrinsic_error

    class Checks:
        def __init__(self, household):
            self.household = household
            self.old_young_conflict = np.array([0, 0.5, 0.25, 0.5])
            self.other_family = np.array([0, 0.5, 0.5, 0.5])
            self.other_dependent_family = np.array([1.5, 0.5, 0.5, 0.5])
            self.single_parent_1 = np.array([1.5, 0.5, 1, 0])
            self.single_parent_2 = np.array([0, 1.5, 1.5, 0])
            self.couple_nd_children = np.array([0, 1.5, 2, 0])

        def perform_checks(self):
            self.check_largest_size()
            if np.array_equal(self.household.composition, self.old_young_conflict): self.check_old_young_conflict()
            elif np.array_equal(self.household.composition, self.other_family): self.check_other_family()
            elif (np.array_equal(self.household.composition, self.single_parent_1)
                  or np.array_equal(self.household.composition, self.single_parent_2)): self.check_single_parent()
            elif np.array_equal(self.household.composition, self.other_dependent_family): self.check_couple_nd_children()

        def check_min_size(self, requirements):
            """Check if the household size meets the minimum required people."""
            if self.household.total_size < self.household.min_size:
                size_deficit = self.household.min_size - self.household.total_size
                requirements['min_required_people'] = size_deficit
                self.household.intrinsic_error += size_deficit * 2
                self.household.accepting_weight = 5

        def check_largest_size(self):
            oversize = self.household.total_size - self.household.TOAH.largest_size
            if oversize > 0:
                if self.household.TOAH.largest_size == 8:
                    self.household.donation_weight = 2 * oversize
                    self.household.intrinsic_error += 0.5 * oversize
                else:
                    self.household.donation_weight = 2 * oversize
                    self.household.intrinsic_error += oversize

        def check_old_young_conflict(self):
            has_adults = self.household.num_people['adult'] > 0
            has_youth = self.household.num_people['youth'] > 0
            has_adults_or_youth = has_adults or has_youth
            has_old = self.household.num_people['old'] > 0

            if has_adults_or_youth and has_old:
                # Conflict: Both old people and adults/youth present
                self.household.intrinsic_error += 100
                self.household.donation_weight += 5

        def check_other_family(self):
            """Check conditions specific to other_family composition."""
            adults = self.household.num_people['adult']
            old_people = self.household.num_people['old']

            if 0 < old_people <= 2 and 0 < adults <= 2:
                # Multigenerational household: one set of adults living with parents
                self.household.intrinsic_error -= 10  # Encourage this configuration
                self.household.accepting_weight += 1
            elif old_people == 0 and adults > 2:
                # Multiple family household: multiple adults, no old people
                self.household.intrinsic_error -= 10  # Encourage multiple family households
                self.household.accepting_weight += 1
            elif old_people >= 1 and adults > 2:
                # Households with multiple adults and old people
                # Less common, so we don't encourage this configuration
                self.household.intrinsic_error += old_people * (adults - 2) * 30  # Penalize to discourage
                self.household.donation_weight += 5
            elif adults == 0:
                self.household.intrinsic_error += 100

        def check_other_dependent_family(self):
            """Check conditions specific to other_dependent_family composition."""
            kids = self.household.num_people['kid']
            adults = self.household.num_people['adult']
            old_people = self.household.num_people['old']

            if kids > 0 and adults > 0 and old_people > 0:
                # Ideal multigenerational household
                self.household.intrinsic_error -= 10  # Encourage this configuration
                if adults <= 2:
                    self.household.intrinsic_error -= 10
                if old_people > 2:
                    excess = old_people - 2
                    self.household.intrinsic_error += 5 * excess ** 2
            else:
                # Penalize if any generation is missing
                missing_generations = sum([
                    kids == 0,
                    adults == 0,
                    old_people == 0
                ])
                self.household.intrinsic_error += 20 * missing_generations
                self.household.donation_weight += missing_generations

            if adults == 0 or adults >= 5:
                self.household.intrinsic_error += 100
                self.household.donation_weight += 5

        def check_single_parent(self):
            """Check conditions specific to single_parent composition."""
            num_kids = self.household.num_people['kid']
            num_youths = self.household.num_people['youth']
            if num_youths == 0:
                if num_kids == 3:
                    # Do not add intrinsic error but increase donation weight
                    self.household.donation_weight += 1  # Slightly more likely to donate
                elif num_kids >= 4:
                    # Add intrinsic error and increase donation weight for kids beyond 3
                    excess_kids = num_kids - 3
                    self.household.intrinsic_error += excess_kids * 3  # Penalty for each kid beyond 3
                    self.household.donation_weight += excess_kids * 3  # More likely to donate
            else:
                if num_youths > 2:
                    self.household.intrinsic_error = num_youths ** 2 * 5
                    self.household.donation_weight += (num_youths - 2) * 2
                children = num_youths + num_kids
                if  children == 3:
                    self.household.donation_weight += 1  # Slightly more likely to donate
                elif children >= 4:
                    excess_children = children - 3
                    self.household.intrinsic_error += (num_youths ** 2) * children * 5  # Penalty for each kid beyond 3
                    self.household.donation_weight += excess_children * 5  # More likely to donate

        def check_couple_nd_children(self):
            num_youths = self.household.num_people['youth']
            if num_youths > 2:
                self.household.intrinsic_error = num_youths ** 2 * 5
                self.household.donation_weight += (num_youths - 2) * 2

        def check_d_children(self):
            num_youths = self.household.num_people['youth']
            if num_youths > 2:
                self.household.intrinsic_error = num_youths ** 2 * 5
                self.household.donation_weight += (num_youths - 2) * 2

    def __repr__(self):
        num_people = [self.num_people[category] for category in type_map.values()]
        return f"id:{self.id}, total_size:{self.total_size}, People: {num_people}"


class MetropolisHastingsAnalysis:
    def __init__(self, household_instance):
        self.household_instance = household_instance

    def _objective_function(self):
        """Calculate how close the current household size distribution is to the target."""
        error = 0
        for i, target_count in enumerate(self.household_instance.var_target_dist):
            current_count = self.household_instance.variable_distribution[i]
            weight = 10 if target_count == 0 else 1 / target_count
            error += 2 * weight * (current_count - target_count) ** 2
        # for i, target_count in enumerate(self.household_instance.)
        #
        return error

    def _weighted_random_choice(self, items, weights):
        """Select an item based on weights."""
        total_weight = sum(weights)
        rnd = random.uniform(0, total_weight)
        upto = 0
        for item, weight in zip(items, weights):
            if upto + weight >= rnd:
                return item
            upto += weight
        return items[-1]

    def _metropolis_hastings_step(self, temperature):
        """Perform one Metropolis-Hastings step."""
        # Create a list of (household, person_type, weight) for donating households
        donating_households = [
            (household, person_type, household.donation_weight)
            for household in self.household_instance.households
            for person_type in type_map.values()
            if household.can_donate(person_type) and household.donation_weight > 0
        ]

        if not donating_households:
            return  # No moves possible

        # Extract households and their weights for weighted selection
        households = [(household, person_type) for household, person_type, _ in donating_households]
        donation_weights = [weight for _, _, weight in donating_households]

        # Select donor household and person type using weighted choice
        donor_household, selected_ptype = self._weighted_random_choice(households, donation_weights)

        # Determine possible accepting person types
        possible_accepting_types = [selected_ptype]

        # If selected_ptype is 'adult' and transition limit not reached, can also consider 'youth'
        if selected_ptype == 'youth' and self.household_instance.transitioned < self.household_instance.transition_population:
            possible_accepting_types.append('adult')

        # If selected_ptype is 'youth' and transitions have occurred, can consider 'adult'
        if selected_ptype == 'adult' and self.household_instance.transitioned > 0:
            possible_accepting_types.append('youth')

        # Combine accepting households for possible accepting types
        accepting_households = []
        for ptype in possible_accepting_types:
            households = self.household_instance.accepting_households.get(ptype, set())
            for household in households:
                accepting_households.append((household, ptype))

        if not accepting_households:
            return  # No valid target household

        # Randomly select an accepting household and target person type
        target_household, target_ptype = random.choice(accepting_households)

        # Calculate current objective value
        current_error = self._objective_function() + self.household_instance.intrinsic_error

        # Store original sizes capped at 8
        old_sizes = np.clip(
            np.array([donor_household.total_size, target_household.total_size]), a_min=None, a_max=8
        )

        # Tentatively move the person with possible transition
        error_change = self.household_instance._donate_person_with_transition(
            donor=donor_household,
            target=target_household,
            original_person_type=selected_ptype,
            target_person_type=target_ptype,
            reverse = False
        )

        if error_change is None:
            return  # Move could not be performed due to transition limits

        # Store new sizes capped at 8
        new_sizes = np.clip(
            np.array([donor_household.total_size, target_household.total_size]), a_min=None, a_max=8
        )

        # Update the current distribution
        for i in range(2):
            self.household_instance.variable_distribution[old_sizes[i]] -= 1
            self.household_instance.variable_distribution[new_sizes[i]] += 1

        # Calculate new objective value
        new_error = self._objective_function() + (self.household_instance.intrinsic_error + error_change)

        # Decide whether to accept or reject the move
        delta_error = new_error - current_error
        if delta_error < 0:
            accept_move = True  # Improvement, always accept
        else:
            acceptance_probability = np.exp(-delta_error / temperature)
            accept_move = random.uniform(0, 1) < acceptance_probability

        # If we reject the move, undo it
        if not accept_move:
            self.household_instance._donate_person_with_transition(
                donor=donor_household,
                target=target_household,
                original_person_type=selected_ptype,
                target_person_type=target_ptype,
                reverse=True
            )

            # Revert the update in current distribution
            for i in range(2):
                self.household_instance.variable_distribution[old_sizes[i]] += 1
                self.household_instance.variable_distribution[new_sizes[i]] -= 1
        else:
            self.household_instance.intrinsic_error += error_change

    def run(self, iterations, initial_temperature, cooling_rate):
        """Run the Metropolis-Hastings process for a number of iterations."""
        temperature = initial_temperature
        for i in range(iterations):
            temperature *= cooling_rate
            self._metropolis_hastings_step(temperature)
            if i == iterations - 1:
                print(f"Iteration {i}:")
                print(f"Total Error = {self._objective_function() + self.household_instance.intrinsic_error}")
                print(f"Number Error = {self._objective_function()}")
                print(f"Intrinsic Error = {self.household_instance.intrinsic_error}")
                print(f"{self.household_instance.transitioned} of possible {self.household_instance.transition_population} have swapped from 'adult' to 'youth'")


class Chi2Analysis:
    def __init__(self, hh_instance, expected_values: pd.DataFrame):
        self.hh_instance = hh_instance
        self.expected_values = expected_values

    @classmethod
    def init(cls, hh_instance, statistics, var_hh_composition):
        # Create DataFrame from the nested dictionary
        df = pd.DataFrame.from_dict(var_hh_composition, orient='index')

        # Sum the 'number' for rows where 'type' is 'd_family'
        d_family_sum = df.loc[df['type'] == 'd_family', 'number'].sum()

        expected_values = d_family_sum * statistics

        return cls(
            hh_instance=hh_instance,
            expected_values=expected_values,
        )

    def analyze_kids_distribution(self):
        one_person_households_kids = []
        two_person_households_kids = []

        # Create a mapping function to categorize the number of kids
        def categorize_kids(num_kids):
            return min(num_kids, 3)  # If num_kids >= 3, it will return 3

        for hh in self.hh_instance.households:
            # Define the relevant compositions for 1-parent and 2-parent households
            composition_map = {
                '[1.5 0.5 1.  0. ]': one_person_households_kids,
                '[1.5 0.5 2.  0. ]': two_person_households_kids,
            }

            # Convert household composition to string to match the dictionary keys
            hh_composition_str = str(hh.composition)

            # Check if the current household's composition is in the map
            if hh_composition_str in composition_map:
                num_kids = hh.num_people['kid']
                categorized_kids = categorize_kids(num_kids)
                composition_map[hh_composition_str].append(categorized_kids)

        # Calculate distributions
        one_person_distribution = pd.Series(one_person_households_kids).value_counts().sort_index()
        two_person_distribution = pd.Series(two_person_households_kids).value_counts().sort_index()

        # Create DataFrames for the distributions for easy comparison later
        self.produced_distribution = pd.DataFrame({
            '1 parent': [one_person_distribution.get(1, 0),
                         one_person_distribution.get(2, 0),
                         one_person_distribution.get(3, 0)],
            '2 parents': [two_person_distribution.get(1, 0),
                          two_person_distribution.get(2, 0),
                          two_person_distribution.get(3, 0)],
        }, index=['1', '2', '3'])
        self.produced_distribution.index.name = 'Number of Kids'

        self.compare_distributions()

        return

    def compare_distributions(self):
        from scipy.stats import chisquare

        # Convert the expected values into a similar format for comparison
        expected_distribution = pd.DataFrame(self.expected_values)
        expected_distribution.index.name = 'Number of Kids'

        # Print both tables side by side for easy comparison
        print("\n--- Produced Distribution ---")
        print(tabulate(self.produced_distribution, headers='keys', tablefmt='pipe', showindex=True))

        print("\n--- Expected Distribution ---")
        print(tabulate(expected_distribution, headers='keys', tablefmt='pipe', showindex=True))

        # Calculate Chi-squared value
        combined_produced_values = []
        combined_expected_values = []
        for parent_type in ['1 parent', '2 parents']:
            produced_values = self.produced_distribution[parent_type].values
            expected_values = expected_distribution[parent_type].values

            # Ensure sums of produced and expected values are equal
            total_produced = sum(produced_values)
            total_expected = sum(expected_values)
            if not np.isclose(total_produced, total_expected, rtol=1e-08):
                adjustment_factor = total_produced / total_expected
                expected_values = expected_values * adjustment_factor

            combined_produced_values.extend(produced_values)
            combined_expected_values.extend(expected_values)

        # Perform Chi-squared test on combined values
        chi2, p = chisquare(combined_produced_values, f_exp=combined_expected_values)
        print(f"\nCombined Chi-squared test: chi2 = {chi2}, p-value = {p}")


        return



