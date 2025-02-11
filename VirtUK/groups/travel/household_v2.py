from collections import defaultdict
import numpy as np
from random import random
from typing import List

from VirtUK.groups import Group, Supergroup
from VirtUK.groups.group.interactive import InteractiveGroup


class GeneralHousehold(Group):
    """
    The GeneralHousehold class stores shared operations among all household types,
    such as managing residents' stay-at-home behavior, basic household structure, etc.
    """
    __slots__ = (
        "area",
        "composition_type",
        "max_size",
        "residents",
        "quarantine_starting_date",
        "residences_to_visit",
        "being_visited",
        "household_to_care",
        "receiving_care",
    )

    def __init__(self, area=None, max_size=np.inf, composition_type=None):
        super().__init__()
        self.area = area
        self.quarantine_starting_date = -99
        self.max_size = max_size
        self.residents = ()
        self.residences_to_visit = defaultdict(tuple)
        self.household_to_care = None
        self.being_visited = False  # this is True when people from other households have been added to the group
        self.receiving_care = False
        self.composition_type = composition_type

    def make_household_residents_stay_home(self, to_send_abroad=None):
        """
        Forces the residents to stay home if they are away doing leisure.
        This is used to welcome visitors.
        """
        for mate in self.residents:
            if mate.busy:
                if mate.leisure is not None:  # this person has already been assigned somewhere
                    if not mate.leisure.external:
                        if mate not in mate.leisure.people:
                            # person active somewhere else, let's not disturb them
                            continue
                        mate.leisure.remove(mate)
                    else:
                        ret = to_send_abroad.delete_person(mate, mate.leisure)
                        if ret:
                            # person active somewhere else, let's not disturb them
                            continue
                    mate.subgroups.leisure = mate.residence
                    mate.residence.append(mate)
            else:
                mate.subgroups.leisure = (
                    mate.residence  # person will be added later in the simulator.
                )

    @property
    def coordinates(self):
        return self.area.coordinates

    @property
    def n_residents(self):
        return len(self.residents)

    def quarantine(self, time, quarantine_days, household_compliance):
        if self.quarantine_starting_date:
            if (
                self.quarantine_starting_date
                < time
                < self.quarantine_starting_date + quarantine_days
            ):
                return random() < household_compliance
        return False

    @property
    def super_area(self):
        try:
            return self.area.super_area
        except AttributeError:
            return None

    def clear(self):
        super().clear()
        self.being_visited = False
        self.receiving_care = False

    def add(self, person, subgroup_type=None, activity="residence"):
        if activity == "residence":
            self.residents = tuple((*self.residents, person))
        else:
            raise NotImplementedError(f"Activity {activity} not supported in household")

# Specific Household Types
class OldHousehold(GeneralHousehold):
    def __init__(self, area=None, max_size=np.inf, composition_type=None):
        super().__init__(area=area, max_size=max_size, composition_type=composition_type)


class NoKidsHousehold(GeneralHousehold):
    def __init__(self, area=None, max_size=np.inf, composition_type=None):
        super().__init__(area=area, max_size=max_size, composition_type=composition_type)


class DependentKidsHousehold(GeneralHousehold):
    def __init__(self, area=None, max_size=np.inf, composition_type=None):
        super().__init__(area=area, max_size=max_size, composition_type=composition_type)


class NoDependentKidsHousehold(GeneralHousehold):
    def __init__(self, area=None, max_size=np.inf, composition_type=None):
        super().__init__(area=area, max_size=max_size, composition_type=composition_type)


class OtherHousehold(GeneralHousehold):
    def __init__(self, area=None, max_size=np.inf, composition_type=None):
        super().__init__(area=area, max_size=max_size, composition_type=composition_type)


# InteractiveHousehold to represent detailed interaction scenarios
class InteractiveHousehold(InteractiveGroup):
    def get_processed_beta(self, betas, beta_reductions):
        """
        In the case of households, we need to apply the beta reduction of household visits
        if the household has a visit, otherwise we apply the beta reduction for a normal
        household.
        """
        if self.group.receiving_care:
            beta = betas["care_visits"]
            beta_reduction = beta_reductions.get("care_visits", 1.0)
        elif self.group.being_visited:
            beta = betas["household_visits"]
            beta_reduction = beta_reductions.get("household_visits", 1.0)
        else:
            beta = betas["household"]
            beta_reduction = beta_reductions.get(self.spec, 1.0)
        regional_compliance = self.super_area.region.regional_compliance
        return beta * (1 + regional_compliance * (beta_reduction - 1))

# Supergroup of Households
class Households(Supergroup):
    """
    Contains all households for the given area and information about them.
    """
    venue_class = GeneralHousehold

    def __init__(self, households: List[GeneralHousehold]):
        super().__init__(members=households)
        self.old = [household for household in households if isinstance(household, OldHousehold)]
        self.no_kids = [household for household in households if isinstance(household, NoKidsHousehold)]
        self.dependent_kids = [household for household in households if isinstance(household, DependentKidsHousehold)]
        self.no_dependent_kids = [household for household in households if isinstance(household, NoDependentKidsHousehold)]
        self.other = [household for household in households if isinstance(household, OtherHousehold)]

    @property
    def households(self):
        return self.members