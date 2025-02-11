import numpy as np
from dataclasses import dataclass

@dataclass
class configs:
    other_student_old = np.array([0, 0, 0.5, 0, 0.5])
    other_family = np.array([0, 0.5, 0.5, 0.5, 0.5])
    other_dependent_family = np.array([1.5, 0.5, 0.5, 0.5, 0.5])
    single_parent_1 = np.array([1.5, 0.5, 0.5, 1, 0])
    single_parent_2 = np.array([0, 1.5, 0.5, 1.5, 0])
    couple_nd_children = np.array([0, 1.5, 0.5, 2, 0])


class Boundary_Conditions:
    def __init__(self, household):
        self.household = household

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


class Intrinsic_Error:
    def __init__(self, household):
        self.household = household
        # Predefined mapping of configurations to methods
        self.check_map = {
            tuple(configs.other_student_old): self.other_student_old,
            tuple(configs.other_family): self.other_family,
            tuple(configs.other_dependent_family): self.other_dependent_family,
            tuple(configs.single_parent_1): self.single_parent,
            tuple(configs.single_parent_2): self.single_parent,
        }

    def perform_checks(self):
        comp_tuple = tuple(self.household.composition)
        check_function = self.check_map.get(comp_tuple, None)
        if check_function:
            check_function()

    def add_intrinsic_error(self, condition, error_value, donation_value=None):
        """
        Helper to add intrinsic error and optional donation weight.
        """
        if condition:
            self.household.intrinsic_error += error_value
            if donation_value is not None:
                self.household.donation_weight += donation_value

    def other_student_old(self):
        has_students = self.household.num_people['student'] > 0
        has_old = self.household.num_people['old'] > 0
        self.add_intrinsic_error(
            has_students and has_old,
            100,
            5)

    def other_family(self):
        adults = self.household.num_people['adult']
        old_people = self.household.num_people['old']

        self.add_intrinsic_error(
            0 < old_people <= 2 and 0 < adults <= 2,
            -10,  # Encourage multigenerational households
            1
        )
        self.add_intrinsic_error(
            old_people == 0 and adults > 2,
            -10,  # Encourage multiple family households
            1
        )
        self.add_intrinsic_error(
            old_people >= 1 and adults > 2,
            old_people * (adults - 2) * 30,  # Penalize for less common configurations
            5
        )
        self.add_intrinsic_error(
            adults == 0,
            100)

    def other_dependent_family(self):
        kids = sum([
            self.household.num_people['kid'],
            self.household.num_people['student'],
            self.household.num_people['youth']
        ])
        adults = self.household.num_people['adult']
        old_people = self.household.num_people['old']

        self.add_intrinsic_error(
            kids > 0 and adults > 0 and old_people > 0,
            -10,  # Encourage ideal multigenerational households
        )
        self.add_intrinsic_error(
            adults <= 2,
            -10)
        self.add_intrinsic_error(
            old_people > 2,
            5 * (old_people - 2) ** 2)

        missing_generations = sum([
            kids == 0,
            adults == 0,
            old_people == 0
        ])
        self.add_intrinsic_error(
            missing_generations > 0,
            20 * missing_generations,
            missing_generations)
        self.add_intrinsic_error(
            adults == 0 or adults >= 5,
            100,
            5)

    def single_parent(self):
        num_kids = self.household.num_people['kid']
        num_youths = self.household.num_people['youth'] + self.household.num_people['student']
        children = num_youths + num_kids

        self.add_intrinsic_error(
            num_youths == 0 and num_kids == 3,
            0,  # No intrinsic error, slightly increase donation weight
            1
        )
        self.add_intrinsic_error(
            num_youths == 0 and num_kids >= 4,
            (num_kids - 3) * 3,  # Penalize for each kid beyond 3
            (num_kids - 3) * 3
        )
        self.add_intrinsic_error(
            num_youths > 2,
            num_youths ** 2 * 5,
            (num_youths - 2) * 2
        )
        self.add_intrinsic_error(
            children == 3,
            0,
            1  # Slightly more likely to donate
        )
        self.add_intrinsic_error(
            children >= 4,
            num_youths ** 2 * children * 5,  # Penalty for excess kids
            (children - 3) * 5
        )

    def couple_nd_children(self):
        num_youths = self.household.num_people['youth'] + self.household.num_people['student']
        self.add_intrinsic_error(
            num_youths > 2,
            num_youths ** 2 * 5,
            (num_youths - 2) * 2)
