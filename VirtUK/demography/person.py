from itertools import count
from random import choice
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Any

# from VirtUK.epidemiology.infection import Infection, Immunity

if TYPE_CHECKING:
    from VirtUK.geography.geography import Area, SuperArea
    from VirtUK.groups.travel.mode_of_transport import ModeOfTransport
    # from VirtUK.policy.vaccine_policy import VaccineTrajectory

HOSPITAL = "hospital"
CARE_HOME = "care_home"
PATIENTS = "patients"
ICU_PATIENTS = "icu_patients"

@dataclass
class Activities:
    residence: Optional[str] = None
    primary_activity: Optional[str] = None
    medical_facility: Optional[str] = None
    commute: Optional[str] = None
    rail_travel: Optional[str] = None
    leisure: Optional[str] = None

    def iter(self):
        return [getattr(self, activity) for activity in self.__annotations__]


person_ids = count()


@dataclass
class Person:
    id: int
    sex: str
    age: int
    partnership_status: Optional[str] = None
    spouse: Optional["Person"] = None
    offspring: Optional[List["Person"]] = None
    ethnicity: Optional[str] = None

    area: Optional["Area"] = None
    work_super_area: Optional["SuperArea"] = None
    sector: Optional[str] = None
    sub_sector: Optional[str] = None
    lockdown_status: Optional[str] = None
    # vaccine_trajectory: Optional["VaccineTrajectory"] = None
    vaccinated: Optional[int] = None
    vaccine_type: Optional[str] = None
    comorbidity: Optional[str] = None
    mode_of_transport: Optional["ModeOfTransport"] = None
    busy: bool = False
    subgroups: Activities = field(default_factory=Activities)
    # infection: Optional[Infection] = None
    # immunity: Optional[Immunity] = None
    dead: bool = False

    @classmethod
    def from_attributes(
        cls,
        sex: str,
        age: int,
        susceptibility_dict: Optional[dict] = None,
        ethnicity: Optional[str] = None,
        partnership_status: Optional[str] = None,
        id: Optional[int] = None,
        comorbidity: Optional[str] = None,
    ) -> "Person":
        if id is None:
            id = next(person_ids)
        return cls(
            id=id,
            sex=sex,
            age=age,
            ethnicity=ethnicity,
            partnership_status=partnership_status,
            # immunity=Immunity(susceptibility_dict=susceptibility_dict),
            comorbidity=comorbidity,
        )

    @property
    def infected(self) -> bool:
        return self.infection is not None

    @property
    def residence(self):
        return self.subgroups.residence

    @property
    def primary_activity(self):
        return self.subgroups.primary_activity

    @property
    def medical_facility(self):
        return self.subgroups.medical_facility

    @property
    def commute(self):
        return self.subgroups.commute

    @property
    def rail_travel(self):
        return self.subgroups.rail_travel

    @property
    def leisure(self):
        return self.subgroups.leisure

    @property
    def hospitalised(self) -> bool:
        group = getattr(self.medical_facility, 'group', None)
        return (
            group and group.spec == HOSPITAL
            and getattr(self.medical_facility, 'subgroup_type', None) == getattr(group, 'SubgroupType', {}).get(PATIENTS)
        )

    @property
    def intensive_care(self) -> bool:
        group = getattr(self.medical_facility, 'group', None)
        return (
            group and group.spec == HOSPITAL
            and getattr(self.medical_facility, 'subgroup_type', None) == getattr(group, 'SubgroupType', {}).get(ICU_PATIENTS)
        )

    @property
    def housemates(self):
        if getattr(self.residence, 'group', {}).get('spec') == CARE_HOME:
            return []
        return getattr(self.residence, 'group', {}).get('residents', [])

    def find_guardian(self):
        def is_eligible_guardian(person):
            return person.age >= 18 and not person.dead and not (person.infection and person.infection.should_be_in_hospital)

        possible_guardians = [person for person in self.housemates if is_eligible_guardian(person)]
        return choice(possible_guardians) if possible_guardians else None

    @property
    def symptoms(self):
        return None if self.infection is None else self.infection.symptoms

    @property
    def super_area(self):
        return getattr(self.area, 'super_area', None)

    @property
    def lad(self):
        return getattr(self.area, 'lad', None)

    @property
    def region(self):
        return getattr(self.super_area, 'region', None)

    @property
    def home_city(self):
        return getattr(self.area.super_area, 'city', None)

    @property
    def work_city(self):
        return None if self.work_super_area is None else getattr(self.work_super_area, 'city', None)

    @property
    def available(self) -> bool:
        return not self.dead and self.medical_facility is None and not self.busy

    @property
    def socioeconomic_index(self):
        return getattr(self.area, 'socioeconomic_index', None)
