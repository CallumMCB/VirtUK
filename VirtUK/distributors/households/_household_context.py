from collections import defaultdict
import pandas as pd
import numpy as np

class HouseholdContext:
    def __init__(self):
        self.area = None
        self.msoa = None
        self.lad = None

        self.all_households = []

        self.women_by_age = defaultdict(list)
        self.men_by_age = defaultdict(list)
        self.broad_ages = pd.DataFrame()

        self.number_households = pd.DataFrame
        self.size_households = np.ndarray([])
        self.student_accommodation = pd.DataFrame

        self.partnerships_oa = pd.DataFrame()
        self.partnerships_lad = pd.DataFrame()

        self.dependent_kids_lad = pd.DataFrame()