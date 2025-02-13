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

t1 = time.time()

# default config
config_path = "./config_simulation.yaml"

# define geography, let's run the first 20 super areas of london
geography = Geography.from_file({"lad": ['County Durham']})

'''
# add buildings
geography.companies = Companies.for_geography(geography)
geography.schools = Schools.for_geography(geography)
geography.universities = Universities.for_geography(geography)
geography.care_homes = CareHomes.for_geography(geography)
# generate worldf
'''
world = generate_world_from_geography(geography, include_residences=True)

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
