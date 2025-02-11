import logging
from itertools import count, chain
from typing import List, Dict, Tuple, Optional
import pandas as pd
import json
import numpy as np
from dataclasses import dataclass
from sklearn.neighbors import BallTree

from VirtUK import paths
from VirtUK.demography.person import Person

logger = logging.getLogger(__name__)
earth_radius = 6371  # km

@dataclass(frozen=True)
class GeographyFPs:
    geography_dir = f'{paths.data_path}/input/geography/'

    hierarchy_fp = (geography_dir + 'oa_msoa_lad_regions.csv')
    oa_coordinates_fp = (geography_dir + 'oa_coordinates.csv')
    msoa_coordinates_fp = (geography_dir + 'msoa_coordinates.csv')
    lad_borders_fp = (geography_dir + 'lad_borders.json')
    # socioeconomic_index needs to be added
    socioeconomic_index_fp = None


class GeographyError(BaseException):
    pass


class Area:
    """
    Fine geographical resolution.
    """

    __slots__ = (
        "people",
        "id",
        "name",
        "coordinates",
        "super_area",
        "lad",
        "region",
        "care_home",
        "schools",
        "households",
        "social_venues",
        "socioeconomic_index",
    )
    _id = count()

    def __init__(
        self,
        name: str = None,
        super_area: "SuperArea" = None,
        lad: "LAD" = None,
        coordinates: Tuple[float, float] = None,
        socioeconomic_index: float = None,
    ):
        """
        Coordinate is given in the format [Y, X] where X is longitude and Y is latitude.
        """
        self.id = next(self._id)
        self.name = name
        self.care_home = None
        self.coordinates = coordinates
        self.super_area = super_area
        self.lad = lad
        self.region = super_area.region
        self.socioeconomic_index = socioeconomic_index
        self.people = []
        self.schools = []
        self.households = []
        self.social_venues = {}

    def add(self, person: Person):
        self.people.append(person)
        person.area = self

    def populate(self, demography, ethnicity=True, comorbidity=False):
        for person in demography.populate(
            self.name, ethnicity=ethnicity, comorbidity=comorbidity
        ):
            self.add(person)


class Areas:
    __slots__ = "members_by_id", "super_area", "ball_tree", "members_by_name"

    def __init__(self, areas: List[Area], super_area=None, ball_tree: bool = True):
        self.members_by_id = {area.id: area for area in areas}
        try:
            self.members_by_name = {area.name: area for area in areas}
        except AttributeError:
            self.members_by_name = None
        self.super_area = super_area
        if ball_tree:
            self.ball_tree = self.construct_ball_tree()
        else:
            self.ball_tree = None

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, index):
        return self.members[index]

    def get_from_id(self, id):
        return self.members_by_id[id]

    def get_from_name(self, name):
        return self.members_by_name[name]

    @property
    def members(self):
        return list(self.members_by_id.values())

    def construct_ball_tree(self):
        all_members = self.members
        coordinates = np.array([np.deg2rad(area.coordinates) for area in all_members])
        ball_tree = BallTree(coordinates, metric="haversine")
        return ball_tree

    def get_closest_areas(self, coordinates, k=1, return_distance=False):
        coordinates = np.array(coordinates)
        if self.ball_tree is None:
            raise GeographyError("Areas initialized without a BallTree")
        if coordinates.shape == (2,):
            coordinates = coordinates.reshape(1, -1)
        if return_distance:
            distances, indcs = self.ball_tree.query(
                np.deg2rad(coordinates), return_distance=return_distance, k=k
            )
            if coordinates.shape == (1, 2):
                all_areas = self.members
                areas = [all_areas[idx] for idx in indcs[0]]
                return areas, distances[0] * earth_radius
            else:
                all_areas = self.members
                areas = [all_areas[idx] for idx in indcs[:, 0]]
                return areas, distances[:, 0] * earth_radius
        else:
            indcs = self.ball_tree.query(
                np.deg2rad(coordinates), return_distance=return_distance, k=k
            )
            all_areas = self.members
            areas = [all_areas[idx] for idx in indcs.flatten()]
            return areas

    def get_closest_area(self, coordinates, return_distance=False):
        if return_distance:
            closest_areas, dists = self.get_closest_areas(
                coordinates, k=1, return_distance=return_distance
            )
            return closest_areas[0], dists[0]
        else:
            return self.get_closest_areas(
                coordinates, k=1, return_distance=return_distance
            )[0]


class SuperArea:
    """
    Coarse geographical resolution.
    A group of areas which make up a larger structure.
    """

    __slots__ = (
        "id",
        "name",
        "city",
        "coordinates",
        "closest_inter_city_station_for_city",
        "lad",
        "region",
        "workers",
        "areas",
        "companies",
        "closest_hospitals",
    )
    external = False
    _id = count()

    def __init__(
        self,
        name: Optional[str] = None,
        areas: List[Area] = None,
        coordinates: Tuple[float, float] = None,
        lad: "LAD" = None,
        region: "Region" = None,
    ):
        self.id = next(self._id)
        self.name = name
        self.city = None
        self.closest_inter_city_station_for_city = {}
        self.coordinates = coordinates
        self.lad = lad
        self.region = region
        self.areas = areas or []
        self.workers = []
        self.companies = []
        self.closest_hospitals = None

    def add_worker(self, person: Person):
        self.workers.append(person)
        person.work_super_area = self

    def remove_worker(self, person: Person):
        self.workers.remove(person)
        person.work_super_area = None

    @property
    def people(self):
        return list(chain.from_iterable(area.people for area in self.areas))

    @property
    def households(self):
        return list(chain.from_iterable(area.households for area in self.areas))


class SuperAreas:
    __slots__ = "members_by_id", "ball_tree", "members_by_name"

    def __init__(self, super_areas: List[SuperArea], ball_tree: bool = True):
        """
        Group to aggregate SuperArea objects.

        Parameters
        ----------
        super_areas
            list of super areas
        ball_tree
            whether to construct a NN tree for the super areas
        """
        self.members_by_id = {super_area.id: super_area for super_area in super_areas}
        try:
            self.members_by_name = {
                super_area.name: super_area for super_area in super_areas
            }
        except AttributeError:
            self.members_by_name = None
        if ball_tree:
            self.ball_tree = self.construct_ball_tree()
        else:
            self.ball_tree = None

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, index):
        return self.members[index]

    def get_from_id(self, id):
        return self.members_by_id[id]

    def get_from_name(self, name):
        return self.members_by_name[name]

    @property
    def members(self):
        return list(self.members_by_id.values())

    def construct_ball_tree(self):
        all_members = self.members
        coordinates = np.array(
            [np.deg2rad(super_area.coordinates) for super_area in all_members]
        )
        ball_tree = BallTree(coordinates, metric="haversine")
        return ball_tree

    def get_closest_super_areas(self, coordinates, k=1, return_distance=False):
        coordinates = np.array(coordinates)
        if self.ball_tree is None:
            raise GeographyError("SuperAreas initialized without a BallTree")
        if coordinates.shape == (2,):
            coordinates = coordinates.reshape(1, -1)
        if return_distance:
            distances, indcs = self.ball_tree.query(
                np.deg2rad(coordinates),
                return_distance=return_distance,
                k=k,
                sort_results=True,
            )
            indcs = chain.from_iterable(indcs)
            all_super_areas = self.members
            super_areas = [all_super_areas[idx] for idx in indcs]
            distances = distances.flatten()
            return super_areas, distances * earth_radius
        else:
            indcs = self.ball_tree.query(
                np.deg2rad(coordinates),
                return_distance=return_distance,
                k=k,
                sort_results=True,
            )
            all_super_areas = self.members
            super_areas = [all_super_areas[idx] for idx in indcs.flatten()]
            return super_areas

    def get_closest_super_area(self, coordinates, return_distance=False):
        if return_distance:
            closest_areas, distances = self.get_closest_super_areas(coordinates, k=1, return_distance=return_distance)
            return closest_areas[0], distances[0]
        else: return self.get_closest_super_areas(coordinates, k=1, return_distance=return_distance)[0]


class ExternalSuperArea:
    """
    This a city that lives outside the simulated domain.
    """

    external = True
    __slots__ = "city", "spec", "id", "domain_id", "coordinates"

    def __init__(self, id, domain_id, coordinates):
        self.city = None
        self.spec = "super_area"
        self.id = id
        self.domain_id = domain_id
        self.coordinates = coordinates


class LAD:
    """Coarse geographical resolution to group Super Areas."""

    __slots__ = ("id", "name", "code", "region", "super_areas", "areas", "super_areas")
    _id = count()

    def __init__(
            self,
            name: Optional[str] = None,
            code: Optional[str] = None,
            areas: List[Area] = None,
            super_areas: List[SuperArea] = None,
            region: "Region" = None,
    ):
        self.id = next(self._id)
        self.name = name
        self.code = code
        self.areas = areas or []
        self.super_areas = super_areas or []
        self.region = region

    @property
    def people(self):
        return list(chain.from_iterable(super_area.people for super_area in self.super_areas))

    @property
    def households(self):
        return list(chain.from_iterable(super_area.households for super_area in self.super_areas))

class LADs:
    __slots__ = ("members_by_id", "members_by_name")

    def __init__(self, lads: List[LAD]):
        """
        Group to aggregate LAD objects.

        Parameters
        ----------
        lads
            list of LADs
        """
        self.members_by_id = {lad.id: lad for lad in lads}
        try: self.members_by_name = {lad.name: lad for lad in lads}
        except AttributeError: self.members_by_name = None

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, index):
        return self.members[index]

    def get_from_id(self, id):
        return self.members_by_id[id]

    def get_from_name(self, name):
        return self.members_by_name[name]

    @property
    def members(self):
        return list(self.members_by_id.values())


    def get_bordering_lads(self):
        """
        TODO: Use the bordering file to return LADs which border this one (and maybe those which border those)
        """
        pass

class Region:
    """
    Coarsest geographical resolution
    """

    __slots__ = ("id", "name", "lads", "super_areas", "policy")
    _id = count()

    def __init__(
        self, name: Optional[str] = None, lads: List[LADs] = None, super_areas: List[SuperAreas] = None
    ):
        self.id = next(self._id)
        self.name = name
        self.lads = lads or []
        self.super_areas = super_areas or []
        self.policy = {
            "regional_compliance": 1.0,
            "lockdown_tier": None,
            "local_closed_venues": set(),
            "global_closed_venues": set(),
        }

    @property
    def people(self):
        return list(
            chain.from_iterable(lad.people for lad in self.lads)
        )

    @property
    def regional_compliance(self):
        return self.policy["regional_compliance"]

    @regional_compliance.setter
    def regional_compliance(self, value):
        self.policy["regional_compliance"] = value

    @property
    def closed_venues(self):
        return self.policy["local_closed_venues"] | self.policy["global_closed_venues"]

    @property
    def households(self):
        return list(
            chain.from_iterable(
                lad.households for lad in self.lads
            )
        )


class Regions:
    __slots__ = "members_by_id", "members_by_name"

    def __init__(self, regions: List[Region]):
        self.members_by_id = {region.id: region for region in regions}
        try:
            self.members_by_name = {region.name: region for region in regions}
        except AttributeError:
            self.members_by_name = None

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, index):
        return self.members[index]

    def get_from_id(self, id):
        return self.members_by_id[id]

    def get_from_name(self, name):
        return self.members_by_name[name]

    @property
    def members(self):
        return list(self.members_by_id.values())


class Geography:
    def __init__(
        self, areas: List[Area], super_areas: List[SuperArea], lads: List[LAD], regions: List[Region]
    ):
        self.areas = areas
        self.super_areas = super_areas
        self.lads = lads
        self.regions = regions
        # possible buildings
        self.households = None
        self.schools = None
        self.hospitals = None
        self.companies = None
        self.care_homes = None
        self.pubs = None
        self.cinemas = None
        self.groceries = None
        self.cemeteries = None
        self.universities = None

    @classmethod
    def _create_areas(
        cls,
        area_coords: pd.DataFrame,
        super_area: str,
        lad: str,
        socioeconomic_indices: pd.Series,
    ) -> List[Area]:
        """
        Applies the _create_area function throughout the area_coords dataframe.
        If area_coords is a series object, then it does not use the apply()
        function as it does not support the axis=1 parameter.

        Parameters
        ----------
        area_coords
            pandas Dataframe with the area name as index and the coordinates
            X, Y where X is longitude and Y is latitude.
        """
        # if a single area is given, then area_coords is a series
        # and we cannot do iterrows()
        if isinstance(area_coords, pd.Series):
            areas = [
                Area(
                    area_coords.name,
                    super_area,
                    lad,
                    area_coords.values,
                    socioeconomic_indices.loc[area_coords.name],
                )
            ]
        else:
            areas = []
            for name, coordinates in area_coords.iterrows():
                areas.append(
                    Area(
                        name,
                        super_area,
                        lad,
                        coordinates=np.array(
                            [coordinates.latitude, coordinates.longitude]
                        ),
                        socioeconomic_index=socioeconomic_indices.loc[name],
                    )
                )
        return areas

    @classmethod
    def _create_super_areas(
            cls,
            super_area_coords: pd.DataFrame,
            area_coords: pd.DataFrame,
            area_socioeconomic_indices: pd.Series,
            lad: "LAD",
            region: "Region",
            hierarchy: pd.DataFrame,
    ) -> List[SuperArea]:
        super_area_hierarchy = hierarchy.reset_index()
        super_area_hierarchy.set_index('msoa', inplace=True)
        total_areas_list, super_areas_list = [], []
        if isinstance(super_area_coords, pd.Series):
            super_areas_list = [
                SuperArea(
                    super_area_coords.name,
                    areas=None,
                    lad=lad,
                    region=region,
                    coordinates=np.array(
                        [super_area_coords.latitude, super_area_coords.longitude]
                    ),
                )
            ]
            areas_df = area_coords.loc[
                super_area_hierarchy.loc[super_area_coords.name, "area"]
            ]
            areas_list = cls._create_areas(
                areas_df, super_areas_list[0], lad, area_socioeconomic_indices
            )
            super_areas_list[0].areas = areas_list
            total_areas_list += areas_list
        else:
            for super_area_name, row in super_area_coords.iterrows():
                super_area = SuperArea(
                    areas=None,
                    name=super_area_name,
                    coordinates=np.array([row.latitude, row.longitude]),
                    lad=lad,
                    region=region
                )
                areas_df = area_coords.loc[super_area_hierarchy.loc[super_area_name, "area"]]
                areas_list = cls._create_areas(
                    areas_df, super_area, lad, area_socioeconomic_indices
                )
                super_area.areas = areas_list
                total_areas_list += list(areas_list)
                super_areas_list.append(super_area)
        return super_areas_list, total_areas_list

    @classmethod
    def _create_lads(
            cls,
            lad_borders: dict,
            lad_hierarchy: pd.DataFrame,
            super_area_coords: pd.DataFrame,
            area_coords: pd.DataFrame,
            area_socioeconomic_indices: pd.Series,
            region: "Region",
            hierarchy: pd.DataFrame,
    ) -> List[LAD]:
        """
        Create LADs and assign super areas to them.

        Parameters
        ----------
        lad_hierarchy: pd.DataFrame
            DataFrame containing LAD names and their associated super areas.
        """
        lads_list, total_super_areas_list, total_areas_list = [], [], []
        for lad_name in lad_hierarchy.index.unique():
            lad_code = lad_hierarchy.loc[lad_name, "lad_code"][0]
            lad = LAD(
                name=lad_name,
                code=lad_code,
                areas=None,
                super_areas=None,
                region=region)
            super_areas_df = super_area_coords.loc[
                lad_hierarchy.loc[lad_name, "msoa"]
            ]
            super_areas_list, areas_list = cls._create_super_areas(
                super_areas_df,
                area_coords,
                area_socioeconomic_indices,
                lad,
                region,
                hierarchy=hierarchy,
            )
            lad.areas = areas_list
            lad.super_areas = super_areas_list
            total_super_areas_list += list(super_areas_list)
            total_areas_list += list(areas_list)
            lads_list.append(lad)

        return lads_list, total_super_areas_list, total_areas_list

    @classmethod
    def create_geographical_units(
            cls,
            hierarchy: pd.DataFrame,
            area_coordinates: pd.DataFrame,
            super_area_coordinates: pd.DataFrame,
            lad_borders_data: dict,
            area_socioeconomic_indices: pd.Series,
            sort_identifiers=True,
    ):
        """
        Create geo-graph of the used geographical units.

        """
        # this method ensures that geo units are ordered by identifier.
        region_hierarchy = hierarchy.reset_index().set_index("region")[["lad", "lad_code", "msoa"]].drop_duplicates()

        total_regions_list, total_areas_list, total_super_areas_list, total_lads_list = [], [], [], []

        for region_name in region_hierarchy.index.unique():

            region = Region(name=region_name, lads=None, super_areas=None)
            lad_hierarchy = region_hierarchy.loc[region_name].reset_index().set_index("lad")[["lad_code", "msoa"]].drop_duplicates()

            lads_list, super_areas_list, areas_list = cls._create_lads(
                lad_borders=lad_borders_data,
                lad_hierarchy=lad_hierarchy,
                super_area_coords=super_area_coordinates,
                area_coords=area_coordinates,
                area_socioeconomic_indices=area_socioeconomic_indices,
                region=region,
                hierarchy=hierarchy,
            )
            region.lads = lads_list
            total_lads_list += lads_list
            total_super_areas_list += super_areas_list
            total_areas_list += areas_list
            total_regions_list.append(region)

        if sort_identifiers:
            total_areas_list = sort_geo_unit_by_identifier(total_areas_list)
            total_super_areas_list = sort_geo_unit_by_identifier(total_super_areas_list)
            total_lads_list = sort_geo_unit_by_identifier(total_lads_list)

        areas = Areas(total_areas_list)
        super_areas = SuperAreas(total_super_areas_list)
        lads = LADs(total_lads_list)
        regions = Regions(total_regions_list)
        logger.info(
            f"There are {len(areas)} areas, "
            + f"{len(super_areas)} super_areas, "
            + f"{len(lads)} LADs, "
            + f"and {len(regions)} regions in the world."
        )
        return areas, super_areas, lads, regions

    @classmethod
    def from_file(
            cls,
            filter_key: Optional[Dict[str, list]] = None,
            hierarchy_filename: str = GeographyFPs.hierarchy_fp,
            area_coordinates_filename: str = GeographyFPs.oa_coordinates_fp,
            super_area_coordinates_filename: str = GeographyFPs.msoa_coordinates_fp,
            lad_borders_filename: str = GeographyFPs.lad_borders_fp,
            area_socioeconomic_index_filename: str = GeographyFPs.socioeconomic_index_fp,
            sort_identifiers=True,
    ) -> "Geography":
        """
        Load data from files and construct classes capable of generating
        hierarchical structure of geographical areas.

        Example usage
        -------------
            ```
            geography = Geography.from_file(filter_key={"region" : "North East"})
            geography = Geography.from_file(filter_key={"super_area" : ["E02005728"]})
            ```
        Parameters
        ----------
        filter_key
            Filter out geo-units which should enter the world.
            At the moment this can only be one of [area, super_area, region, lad]
        hierarchy_filename
            Pandas df file containing the relationships between the different
            geographical units.
        area_coordinates_filename:
            coordinates of the area units
        super_area_coordinates_filename
            coordinates of the super area units
        area_socioeconomic_index_filename
            socioeconomic index of each area
        """
        geo_hierarchy = pd.read_csv(hierarchy_filename)
        areas_coord = pd.read_csv(area_coordinates_filename)
        super_areas_coord = pd.read_csv(super_area_coordinates_filename)
        lad_borders_data = json.load(open(lad_borders_filename, 'r'))

        if filter_key is not None:
            geo_hierarchy = _filtering(geo_hierarchy, filter_key)

        areas_coord = areas_coord.loc[areas_coord.area.isin(geo_hierarchy.area)]
        super_areas_coord = super_areas_coord.loc[
            super_areas_coord.msoa.isin(geo_hierarchy.msoa)
        ].drop_duplicates()

        areas_coord.set_index("area", inplace=True)
        areas_coord = areas_coord[["latitude", "longitude"]]
        super_areas_coord.set_index("msoa", inplace=True)
        super_areas_coord = super_areas_coord[["latitude", "longitude"]]
        geo_hierarchy.set_index("msoa", inplace=True)

        if area_socioeconomic_index_filename:
            area_socioeconomic_df = pd.read_csv(area_socioeconomic_index_filename)
            area_socioeconomic_df = area_socioeconomic_df.loc[
                area_socioeconomic_df.area.isin(geo_hierarchy.area)
            ]
            area_socioeconomic_df.set_index("area", inplace=True)
            area_socioeconomic_index = area_socioeconomic_df["socioeconomic_centile"]
        else:
            area_socioeconomic_index = pd.Series(
                data=np.full(len(areas_coord), None),
                index=areas_coord.index,
                name="socioeconomic_centile",
            )

        # Extract the lad hierarchy
        lad_hierarchy = geo_hierarchy.reset_index().set_index("lad")

        areas, super_areas, lads, regions = cls.create_geographical_units(
            hierarchy=geo_hierarchy,
            area_coordinates=areas_coord,
            super_area_coordinates=super_areas_coord,
            lad_borders_data=lad_borders_data,
            area_socioeconomic_indices=area_socioeconomic_index,
            sort_identifiers=sort_identifiers,
        )
        return cls(areas, super_areas, lads, regions)


def _filtering(data: pd.DataFrame, filter_key: Dict[str, list]) -> pd.DataFrame:
    """
    Filter DataFrame for given geo-unit and it's listed names
    """
    return data[
        data[list(filter_key.keys())[0]].isin(list(filter_key.values())[0]).values
    ]


def sort_geo_unit_by_identifier(geo_units):
    geo_identifiers = [unit.name for unit in geo_units]
    sorted_idx = np.argsort(geo_identifiers)
    first_unit_id = geo_units[0].id
    units_sorted = [geo_units[idx] for idx in sorted_idx]
    # reassign ids
    for i, unit in enumerate(units_sorted):
        unit.id = first_unit_id + i
    return units_sorted