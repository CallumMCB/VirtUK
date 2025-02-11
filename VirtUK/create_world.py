from VirtUK.geography import Geography
from VirtUK.groups import Hospitals, Schools, Companies, CareHomes, Universities
from VirtUK.groups.leisure import (
    Pubs,
    Cinemas,
    Groceries,
    Gyms,
    generate_leisure_for_config,
)
from VirtUK.groups.travel import Travel
from VirtUK.world import generate_world_from_geography
import time

# load london super areas
# london_areas = os.path.join(os.path.dirname(__file__), "london_areas.txt")
durham_areas = ['E02004313', 'E02004314', 'E02004315']
# warwick_areas = ['E02006525', 'E02006528', 'E02006530']
# cambridge_example = ['E02003723', 'E02003720']
# # # add King's cross area for station
# if "E00004734" not in london_areas:
#     london_areas = np.append(london_areas, "E02000187")
#
# # add some people commuting from Cambridge
# london_areas = np.concatenate((london_areas, ["E02003719", "E02003720", "E02003721"]))
#
# add Bath as well to have a city with no stations
# london_areas = np.concatenate(
#     (london_areas, ["E02002988", "E02002989", "E02002990", "E02002991", "E02002992"])
# )

t1 = time.time()

# default config
config_path = "./config_simulation.yaml"

# define geography, let's run the first 20 super areas of london
# geography = Geography.from_file({"lad": ["Northumberland"]})
# geography = Geography.from_file({"msoa": london_areas})
# geography = Geography.from_file({"msoa": cambridge_example})
geography = Geography.from_file({"msoa": durham_areas})
# geography = Geography.from_file({"msoa": warwick_areas})
# geography = Geography.from_file({"region": ["North East"]})

'''
# add buildings
geography.companies = Companies.for_geography(geography)
geography.schools = Schools.for_geography(geography)
geography.universities = Universities.for_geography(geography)
geography.care_homes = CareHomes.for_geography(geography)
# generate world
'''
world = generate_world_from_geography(geography, include_households=True)

# some leisure activities
world.pubs = Pubs.for_geography(geography)
world.cinemas = Cinemas.for_geography(geography)
world.groceries = Groceries.for_geography(geography)
world.gyms = Gyms.for_geography(geography)
leisure = generate_leisure_for_config(world, config_filename=config_path)
leisure.distribute_social_venues_to_areas(
    areas=world.areas, super_areas=world.super_areas
)  # this assigns possible social venues to people.
travel = Travel()
travel.initialise_commute(world)
t2 = time.time()
print(f"Took {t2 -t1} seconds to run.")
# save the world to hdf5 to load it later
world.to_hdf5("tests.hdf5")
print("Done :)")
