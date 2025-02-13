import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import logging

from VirtUK.demography import Person
from VirtUK.geography import Area, LAD
from VirtUK import DataLoader


class CommunalDistributor:
    def __init__(
            self,

        ):
        pass

    @classmethod
    def from_data_loader(
            cls,
            data_loader: DataLoader = DataLoader(),
    )-> "CommunalDistributor":
        all_data = data_loader.load_all_data()
        return cls(
            )