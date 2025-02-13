import pandas as pd
from dataclasses import dataclass
from VirtUK import paths

@dataclass(frozen=True)
class FilePaths:
    # Demographics
    demography_dir = f'{paths.data_path}/input/demography/'

    age_dir = (demography_dir + 'age_dist_2021/')

    ages_total = (age_dir + '1_year_ages_total_oa.csv')
    ages_male = (age_dir + '1_year_ages_male_oa.csv')
    ages_female = (age_dir + '1_year_ages_female_oa.csv')

    partnership_status_dir = (f'{paths.data_path}/raw_data/demography/partnership_status_oa_estimates/')

    divorced_separated_female_fp = (partnership_status_dir + 'divorced_separated_female.csv')
    divorced_separated_male_fp = (partnership_status_dir + 'divorced_separated_male.csv')
    never_partnered_female_fp = (partnership_status_dir + 'never_partnered_female.csv')
    never_partnered_male_fp = (partnership_status_dir + 'never_partnered_male.csv')
    partnered_female_fp = (partnership_status_dir + 'partnered_female.csv')
    partnered_male_fp = (partnership_status_dir + 'partnered_male.csv')
    widowed_female_fp = (partnership_status_dir + 'widowed_female.csv')
    widowed_male_fp = (partnership_status_dir + 'widowed_male.csv')

    output_fp = (demography_dir + 'partnership_status/oa_legal_partnership.pkl')
    simplified_output_fp = (demography_dir + 'partnership_status/oa_legal_partnership_simplified.pkl')


class PartnershipDataFormatter:
    def __init__(self, file_paths: FilePaths = FilePaths()):
        # Load the paths for input files
        self.file_paths = file_paths

    def load_csv16(self, filename):
        sample_df = pd.read_csv(filename, nrows=1, index_col=0)
        dtype_dict = {column: 'uint16' for column in sample_df.columns}
        return pd.read_csv(filename, index_col=0, dtype=dtype_dict)

    def load_partnership_status(self):
        # Load all partnership CSVs
        divorced_separated_female = self.load_csv16(self.file_paths.divorced_separated_female_fp)
        divorced_separated_male = self.load_csv16(self.file_paths.divorced_separated_male_fp)
        never_partnered_female = self.load_csv16(self.file_paths.never_partnered_female_fp)
        never_partnered_male = self.load_csv16(self.file_paths.never_partnered_male_fp)
        partnered_female = self.load_csv16(self.file_paths.partnered_female_fp)
        partnered_male = self.load_csv16(self.file_paths.partnered_male_fp)
        widowed_female = self.load_csv16(self.file_paths.widowed_female_fp)
        widowed_male = self.load_csv16(self.file_paths.widowed_male_fp)

        # Create a dictionary to store the data
        data_dict = {
            ('f', 'divorced_separated'): divorced_separated_female,
            ('f', 'never_partnered'): never_partnered_female,
            ('f', 'partnered'): partnered_female,
            ('f', 'widowed'): widowed_female,
            ('m', 'divorced_separated'): divorced_separated_male,
            ('m', 'never_partnered'): never_partnered_male,
            ('m', 'partnered'): partnered_male,
            ('m', 'widowed'): widowed_male,
        }

        # Create a list to hold all the DataFrames with appropriate MultiIndex
        multiindex_dataframes = []
        for (gender, partnership_status), df in data_dict.items():
            df = df.copy()
            df.index.name = 'area'
            df['gender'] = gender
            df['partnership_status'] = partnership_status
            df = df.set_index(['gender', 'partnership_status'], append=True)
            multiindex_dataframes.append(df)

        # Concatenate all DataFrames along the rows (axis=0) and set appropriate index names
        final_df = pd.concat(multiindex_dataframes)
        final_df = final_df.reorder_levels(['area', 'gender', 'partnership_status'])
        final_df = final_df.sort_index()

        return final_df

    def save_final_dataframe(self, df):
        filename = self.file_paths.output_fp
        # Save the final dataframe as a pickle file
        df.to_pickle(filename)

    def simplify_and_save_dataframe(self, df):
        # Simplify the dataframe by merging specific columns
        columns_to_merge = ['16-24', '25-34', '35-49', '50-64']
        df['16-64'] = df[columns_to_merge].sum(axis=1)
        df_simplified = df.drop(columns=columns_to_merge)

        # Rename the '65+' column to '65-99' and reorder columns
        if '65+' in df_simplified.columns:
            df_simplified.rename(columns={'65+': '65-99'}, inplace=True)
        column_order = ['16-64', '65-99'] + [col for col in df_simplified.columns if col not in ['16-64', '65-99']]
        df_simplified = df_simplified[column_order]

        # Save the simplified dataframe as a separate pickle file
        simplified_filename = self.file_paths.simplified_output_fp
        df_simplified.to_pickle(simplified_filename)


# Usage example:
adjuster = PartnershipDataFormatter()
final_df = adjuster.load_partnership_status()
adjuster.save_final_dataframe(final_df)
adjuster.simplify_and_save_dataframe(final_df)
