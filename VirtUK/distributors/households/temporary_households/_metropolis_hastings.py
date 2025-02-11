import random
import numpy as np

type_map = {0: "kid", 1: "youth", 2: "student", 3: "adult", 4: "old"}
rvrs_map = {"kid": 0, "youth": 1, "student": 2, "adult": 3, "old": 4}

class MetropolisHastingsAnalysis:
    def __init__(self, household_instance):
        self.household_instance = household_instance
        self.student_composition_type_map = {"parents": ['[1.5 0.5 0.5 2.  0. ]', '[0.  1.5 0.5 2.  0. ]'],
                                     "other": ['[0.  0.5 0.5 0.5 0.5]'],
                                     "all students": ['[0.  0.  0.5 0.  0.5]']}

        valid_student_compositions = set(self.household_instance.composition_totals.index)
        # Filter the composition_type_map
        self.student_composition_type_map = {
            key: value
            for key, value in self.student_composition_type_map.items()
            if key in valid_student_compositions
        }

    def _objective_function_sizes(self):
        """Calculate how close the current household size distribution is to the target."""
        error = 0
        for i, target_count in enumerate(self.household_instance.var_target_dist):
            current_count = self.household_instance.variable_distribution[i]
            weight = 10 if target_count == 0 else 1 / target_count
            error += weight * (current_count - target_count) ** 2
        return error

    def _objective_function_students(self):
        """Calculate how close the current student distribution is to the target."""
        error = 0
        for key, target_count in ((k, v) for k, v in self.household_instance.students_target.items() if
                                  k in self.student_composition_type_map):
            current_count = 0
            for composition in self.student_composition_type_map[key]:
                current_count += self.household_instance.composition_totals.loc[composition, 'student']
            weight = 10 if target_count == 0 else 1 / target_count
            error += weight * (current_count - target_count) ** 2
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
        current_error = self._objective_function_sizes() + self._objective_function_students() + self.household_instance.intrinsic_error

        # Store original sizes capped at 8
        old_sizes = np.clip(
            np.array([donor_household.total_size, target_household.total_size]), a_min=None, a_max=8
        )

        # Tentatively move the person with possible transition
        intrinsic_error_change = self.household_instance._donate_person_with_transition(
            donor=donor_household,
            target=target_household,
            original_person_type=selected_ptype,
            target_person_type=target_ptype,
            reverse = False
        )

        if intrinsic_error_change is None:
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
        new_error = self._objective_function_sizes() + self._objective_function_students() + (self.household_instance.intrinsic_error + intrinsic_error_change)

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
            self.household_instance.intrinsic_error += intrinsic_error_change

    def run(self, iterations, initial_temperature, cooling_rate):
        """Run the Metropolis-Hastings process for a number of iterations."""
        temperature = initial_temperature
        for i in range(iterations):
            temperature *= cooling_rate
            self._metropolis_hastings_step(temperature)
            if i == iterations - 1:
                print(f"Iteration {i}:")
                print(f"Total Error = {self._objective_function_sizes() +self._objective_function_students() + self.household_instance.intrinsic_error}")
                print(f"Size Error = {self._objective_function_sizes()}")
                print(f"Student Error = {self._objective_function_students()}")
                print(f"Intrinsic Error = {self.household_instance.intrinsic_error}")
                print(f"{self.household_instance.transitioned} of possible {self.household_instance.transition_population} have swapped from 'youth' to 'adult'")