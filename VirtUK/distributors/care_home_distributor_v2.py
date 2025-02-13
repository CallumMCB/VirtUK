import logging
import numpy as np
import pandas as pd

from dataclasses import dataclass

from VirtUK import paths
from VirtUK.geography import Area, SuperAreas


logger = logging.getLogger("care_home_distributor")

@dataclass(frozen=True)
class FilePaths:
    households_dir = f"{paths.data_path}/input/households/"
    communal_establishments_dir = (households_dir + '/communal_establishments/')

    care_homes = (communal_establishments_dir + 'care_homes_beds_type_locations.csv')
    oa_tot_com_res = (communal_establishments_dir + 'communal_residents_oa.csv')

    msoa_f_residents = (communal_establishments_dir + 'female_residents_msoa.csv')
    msoa_m_residents = (communal_establishments_dir + 'male_residents_msoa.csv')

    msoa_type_resident = (communal_establishments_dir + 'resident_type_msoa.csv')
    msoa_staff_temporary = (communal_establishments_dir + 'staff_or_temporary_msoa.csv')

    oa_student_accom = (communal_establishments_dir + 'student_accommodation.csv')

default_config_filename = paths.configs_path / "defaults/groups/care_home.yaml"

class DataLoader:
    def __init__(
            self,
            file_paths: FilePaths = FilePaths(),
    ):
        self.file_paths = file_paths

    @staticmethod
    def load_csv16(filename):
        sample_df = pd.read_csv(filename, nrows=1, index_col=0)
        dtype_dict = {column: 'uint16' for column in sample_df.columns}
        return pd.read_csv(filename, index_col=0, dtype=dtype_dict)

    def load_care_homes(self):
        # Read the CSV, selecting and renaming columns in one step
        file_path = self.file_paths.care_homes
        df = pd.read_csv(file_path,
                         usecols=['OA Code', 'Care homes beds', 'Service type - Care home service with nursing'])
        df.rename(columns={'OA Code': 'area', 'Care homes beds': 'beds',
                           'Service type - Care home service with nursing': 'nursing'}, inplace=True)

        # Assign unique IDs and set index
        df['ID'] = range(1, len(df) + 1)
        df.set_index('area', inplace=True)

        # Convert 'nursing' column to boolean
        df['nursing'] = df['nursing'] == 'Y'

        # Reorder columns
        return df[['ID', 'beds', 'nursing']]

    def load_student_accommodation(self):

        # Load the processed data
        df = pd.read_csv(file_path, dtype={col: 'uint16' for col in pd.read_csv(file_path, nrows=1).columns if
                                             col not in ['areas', 'accommodation type']})

        # Set multi-index by 'areas' and 'accommodation type'
        df.set_index(['areas', 'accommodation type'], inplace=True)

        return df



dataloader = DataLoader()
processed_data = dataloader.load_care_homes()
print(processed_data.head(50))

class CareHomeError(BaseException):
    pass


