import numpy as np
from . import Boundary_Conditions, Intrinsic_Error

type_map = {0: "kid", 1: "youth", 2: "student", 3: "adult", 4: "old"}
rvrs_map = {"kid": 0, "youth": 1, "student": 2, "adult": 3, "old": 4}

class Variable_Household:
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
        "accepting_weight",
        "conditions",
        "i_error",
        "TOAH"
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

        self.conditions = Boundary_Conditions(self)
        self.i_error = Intrinsic_Error(self)
        self.TOAH = parent # Temporary Output Area Households

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
            if person_type == 'youth':
                current_num = self.num_people['youth'] + self.num_people['student']
                if  current_num < minimum:
                    required[person_type] = minimum - current_num
            elif self.num_people[person_type] < minimum:
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
        self.conditions.check_min_size(requirements)
        self.conditions.check_largest_size()
        self.i_error.perform_checks()

        # Update conditions satisfied
        self.conditions_satisfied = not requirements
        return requirements, self.intrinsic_error - original_intrinsic_error

    def __repr__(self):
        num_people = [self.num_people[category] for category in type_map.values()]
        return f"id:{self.id}, total_size:{self.total_size}, People: {num_people}"