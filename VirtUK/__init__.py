import logging.config
import os
import yaml

from VirtUK import paths
from .data_presets import FilePaths, DataLoader, AgeLimits
from . import demography
from . import distributors
from . import groups
from .demography import Person
from .world import World

default_logging_config_filename = paths.configs_path / "logging.yaml"

if os.path.isfile(default_logging_config_filename):
    with open(default_logging_config_filename, "rt") as f:
        log_config = yaml.safe_load(f.read())
        logging.config.dictConfig(log_config)
else:
    print("The logging config file does not exist.")
    log_file = os.path.join("./", "world_creation.log")
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
