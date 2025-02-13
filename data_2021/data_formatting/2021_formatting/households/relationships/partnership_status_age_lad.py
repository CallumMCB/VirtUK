import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from VirtUK import paths
from tqdm import tqdm
import os
import pickle
from scipy.interpolate import make_interp_spline

@dataclass
class FilePathConfig:
    hierarchy_file: str = f'{paths.data_path}/input/geography/oa_msoa_lad_regions.csv'
    male_age_file: str = f'{paths.data_path}/input/demography/age_dist_2021/1_year_ages_male_oa.csv'
    female_age_file: str = f'{paths.data_path}/input/demography/age_dist_2021/1_year_ages_female_oa.csv'
    male_partnership_file: str = f'{paths.data_path}/input/demography/partnership_status/lad/m_legal_partnership.csv'
    female_partnership_file: str = f'{paths.data_path}/input/demography/partnership_status/lad/f_legal_partnership.csv'
    output_male_fractions_file: str = f'{paths.data_path}/input/demography/partnership_status/lad/male_legal_partnership_fractions.csv'
    output_female_fractions_file: str = f'{paths.data_path}/input/demography/partnership_status/lad/female_legal_partnership_fractions.csv'
    interpolation_output_file: str = f'{paths.data_path}/input/demography/partnership_status/lad/all_partnership_interpolations.pkl'


class DataProcessor:
    def __init__(self, file_paths: FilePathConfig):
        self.file_paths = file_paths
        self.hierarchy_df = pd.read_csv(self.file_paths.hierarchy_file)
        self.hierarchy_df['lad'] = self.hierarchy_df['lad'].apply(lambda x: x.split(',')[0].strip())
        self.hierarchy_df.set_index('lad', inplace=True)
        self.male_ages_df = pd.read_csv(self.file_paths.male_age_file)
        self.female_ages_df = pd.read_csv(self.file_paths.female_age_file)
        self.male_partnership_df = pd.read_csv(self.file_paths.male_partnership_file)
        self.female_partnership_df = pd.read_csv(self.file_paths.female_partnership_file)
        self.missing_lads = []
        self.lad_name_mapping = {
            'Allerdale': 'Cumberland',
            'Barrow-in-Furness': 'Westmorland and Furness',
            'Carlisle': 'Cumberland',
            'Copeland': 'Cumberland',
            'Craven': 'North Yorkshire',
            'Eden': 'Westmorland and Furness',
            'Hambleton': 'North Yorkshire',
            'Harrogate': 'North Yorkshire',
            'Mendip': 'Somerset',
            'Richmondshire': 'North Yorkshire',
            'Ryedale': 'North Yorkshire',
            'Scarborough': 'North Yorkshire',
            'Sedgemoor': 'Somerset',
            'Selby': 'North Yorkshire',
            'Somerset West and Taunton': 'Somerset',
            'South Lakeland': 'Westmorland and Furness',
            'South Somerset': 'Somerset'
        }
        self.apply_lad_name_mapping()
        # Ensure the directory exists for the output file, not the file itself
        os.makedirs(os.path.dirname(self.file_paths.interpolation_output_file), exist_ok=True)
        self.interpolated_data = pd.DataFrame()

    def apply_lad_name_mapping(self):
        # Apply the mapping to male and female partnership DataFrames
        self.male_partnership_df['lad'] = self.male_partnership_df['lad'].replace(self.lad_name_mapping)
        self.female_partnership_df['lad'] = self.female_partnership_df['lad'].replace(self.lad_name_mapping)

        # Merge duplicate local authorities after applying the mapping
        self.male_partnership_df = self.male_partnership_df.groupby(['lad', 'age_bracket']).sum().reset_index()
        self.female_partnership_df = self.female_partnership_df.groupby(['lad', 'age_bracket']).sum().reset_index()

    def process_all_local_authorities(self):
        all_male_fractions = []
        all_female_fractions = []

        local_authorities = self.male_partnership_df['lad'].unique()
        for i, local_authority_name in enumerate(tqdm(local_authorities, desc="Processing Local Authorities")):
            try:
                corrected_name = self.lad_name_mapping.get(local_authority_name, local_authority_name)
                male_population_by_age, female_population_by_age = self.get_population_by_age(corrected_name)
                male_5y_lad, age_labels = self.get_age_bracket_sums(male_population_by_age)
                female_5y_lad, _ = self.get_age_bracket_sums(female_population_by_age)
                male_partnership, female_partnership = self.get_partnership_data(local_authority_name)

                male_fractions = self.calculate_fractions(male_partnership, male_5y_lad, age_labels)
                female_fractions = self.calculate_fractions(female_partnership, female_5y_lad, age_labels)

                male_fractions.insert(0, 'lad', local_authority_name)
                female_fractions.insert(0, 'lad', local_authority_name)

                all_male_fractions.append(male_fractions)
                all_female_fractions.append(female_fractions)
                if i == 0:
                    self.x_data = np.array(
                        [((int(label.split('-')[0]) + int(label.split('-')[1])) / 2) for label in age_labels[:-1]] + [
                            90])  # Extract integer years from labels
                    self.x_smooth = np.arange(self.x_data.min()-1.5, 100).astype(np.uint8)
                self.perform_interpolation_and_store(local_authority_name, male_fractions, female_fractions, age_labels)

            except KeyError:
                self.missing_lads.append(local_authority_name)

        # Concatenate all fractions and save to new files
        all_male_fractions_df = pd.concat(all_male_fractions).reset_index(drop=True)
        all_female_fractions_df = pd.concat(all_female_fractions).reset_index(drop=True)

        all_male_fractions_df.to_csv(self.file_paths.output_male_fractions_file, index=False)
        all_female_fractions_df.to_csv(self.file_paths.output_female_fractions_file, index=False)

        # Save all interpolated data to one pickle file
        with open(self.file_paths.interpolation_output_file, 'wb') as f:
            pickle.dump(self.interpolated_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.missing_lads:
            print("The following local authorities were missing and could not be processed:")
            for lad in self.missing_lads:
                print(f"- {lad}")

    def perform_interpolation_and_store(self, local_authority_name, male_fractions, female_fractions, age_labels):
        categories = ['never_partnered', 'partnered', 'widowed', 'divorced_separated']
        dataframes = []

        for gender, fractions in [('m', male_fractions), ('f', female_fractions)]:
            interpolated_category_data = []

            # Interpolate data for each category
            for category in categories:
                y_data = fractions[category].values
                spline = make_interp_spline(self.x_data, y_data, k=3, bc_type='natural')
                y_smooth = spline(self.x_smooth)
                y_smooth = np.clip(y_smooth, 0, 1)
                interpolated_category_data.append(y_smooth)

            # Ensure that for each age the sum across all categories is 1
            interpolated_category_data = np.array(interpolated_category_data)
            total_sum = np.sum(interpolated_category_data, axis=0)
            normalized_category_data = np.divide(interpolated_category_data, total_sum)

            # Store the normalized results as a DataFrame with MultiIndex
            df = pd.DataFrame(normalized_category_data, index=categories, columns=self.x_smooth)
            df.index.name = 'category'
            df.columns.name = 'age'

            # Add gender and local authority as multi-index
            df = df.reset_index()
            df.insert(0, 'gender', gender)
            df.insert(0, 'local_authority', local_authority_name)
            df.set_index(['local_authority', 'gender', 'category'], inplace=True)

            dataframes.append(df)

        # Concatenate all DataFrames for both genders for the given local authority
        final_df = pd.concat(dataframes)

        self.interpolated_data = pd.concat([self.interpolated_data, final_df])

    def get_output_areas(self, local_authority_name):
        return self.hierarchy_df.loc[local_authority_name, 'area']

    def get_population_by_age(self, local_authority_name):
        output_areas = self.get_output_areas(local_authority_name)
        male_population = self.male_ages_df[self.male_ages_df['area'].isin(output_areas)].drop(columns=['all_ages'])
        female_population = self.female_ages_df[self.female_ages_df['area'].isin(output_areas)].drop(columns=['all_ages'])
        male_population_sum = male_population.drop(columns=['area']).sum()
        female_population_sum = female_population.drop(columns=['area']).sum()
        return male_population_sum, female_population_sum

    def get_age_bracket_sums(self, population_sum):
        age_bins = [16] + list(range(20, 95, 5))
        population_5y = population_sum.groupby(pd.cut(population_sum.index.astype(int), bins=age_bins, right=False, labels=False)).sum()
        age_labels = ['16-19'] + [f"{i*5 + 20}-{i*5 + 24}" for i in range(13)] + ['85+']
        population_5y.index = age_labels
        return population_5y, age_labels

    def get_partnership_data(self, local_authority_name):
        male_partnership = self.male_partnership_df[self.male_partnership_df['lad'] == local_authority_name]
        female_partnership = self.female_partnership_df[self.female_partnership_df['lad'] == local_authority_name]
        return male_partnership, female_partnership

    def calculate_fractions(self, partnership_data, population_by_age, age_labels):
        categories = ['never_partnered', 'partnered', 'widowed', 'divorced_separated']

        fractions = []
        for age_bracket in population_by_age.index:
            total_population = population_by_age[age_bracket]
            if total_population == 0:
                fractions.append([0] * len(categories))
                continue
            age_data = partnership_data[partnership_data['age_bracket'] == age_bracket].fillna(0)
            fraction = []
            for category in categories:
                value = age_data[category].values[0] if len(age_data) > 0 else 0
                fraction_value = value / total_population if total_population > 0 else 0
                fraction.append(fraction_value)

            total_fraction = sum(fraction)
            if total_fraction > 0:
                fraction = [f / total_fraction for f in fraction]
            fractions.append(fraction)
        return pd.DataFrame(fractions, columns=categories, index=age_labels)


class DataPlotter:
    def __init__(self, file_paths: FilePathConfig):
        self.file_paths = file_paths

    def plot_partnership_status(self, local_authority_name):
        male_fractions_file = self.file_paths.output_male_fractions_file
        female_fractions_file = self.file_paths.output_female_fractions_file
        interpolation_file = self.file_paths.interpolation_output_file
        if not os.path.exists(interpolation_file):
            print(f"Fraction data not found. Please run the data processing step first.")
            return

        if not (os.path.exists(male_fractions_file) and os.path.exists(female_fractions_file)):
            print(f"Fraction data for {local_authority_name} not found. Please run the data processing step first.")
            return

        male_fractions_df = pd.read_csv(male_fractions_file)
        female_fractions_df = pd.read_csv(female_fractions_file)

        with open(interpolation_file, 'rb') as f:
            interpolated_data = pickle.load(f)

        if local_authority_name not in interpolated_data.index:
            print(f"Local authority '{local_authority_name}' not found in the interpolated data.")
            return

        if local_authority_name not in male_fractions_df['lad'].values:
            print(f"Local authority '{local_authority_name}' not found in the data.")
            return

        male_fractions = male_fractions_df[male_fractions_df['lad'] == local_authority_name].drop(columns=['lad'])
        female_fractions = female_fractions_df[female_fractions_df['lad'] == local_authority_name].drop(columns=['lad'])

        age_labels = [16] + list(np.arange(1, 15) * 5 + 15)
        x_positions = [16.1] + list(np.arange(1, len(age_labels)) * 5 + 15.1)  # Set x positions starting from 15, spaced by 5

        fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        self._plot_gender_partnership(ax[0], male_fractions, age_labels, x_positions, "Male")
        self._plot_gender_partnership(ax[1], female_fractions, age_labels, x_positions, "Female")

        ax[-1].set_xlabel("Age Bracket")
        plt.xticks(x_positions, [label for label in age_labels],
                   rotation=45)  # Shift labels to align with center of bars
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        self.plot_uncertainties_with_smooth_curve(male_fractions, female_fractions, age_labels, interpolated_data, local_authority_name)

    def _plot_gender_partnership(self, ax, fractions_df, age_labels, x_positions, title):
        categories = ['never_partnered', 'partnered', 'widowed', 'divorced_separated']
        bottom = np.zeros(len(age_labels))
        bar_widths = [3.8] + [4.8] * 13 + [9.8]

        for j, category in enumerate(categories):
            ax.bar(x_positions, fractions_df[category], width=bar_widths, label=category, bottom=bottom, alpha=0.7, align='edge')
            bottom += fractions_df[category]

        ax.set_title(f"{title} Population by Partnership Status")
        ax.set_ylabel("Fraction of Population")
        ax.legend(loc='center left')

    def plot_uncertainties_with_smooth_curve(self, male_fractions, female_fractions, age_labels, interpolated_data, local_authority_name):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
        categories = ['never_partnered', 'partnered', 'widowed', 'divorced_separated']

        # Extend age labels to include 100 for extrapolation
        age_labels = age_labels + [95]
        x_data = np.array([(age_labels[i] + age_labels[i + 1]) / 2 for i in range(len(age_labels) - 1)])
        xerrors = np.array([(age_labels[i + 1] - age_labels[i]) / 2 for i in range(len(age_labels) - 1)])

        colors = ['blue', 'orange', 'green', 'red']

        for i, category in enumerate(categories):
            # Plot the original data points for both male and female
            y_data_male = male_fractions[category].values
            y_data_female = female_fractions[category].values

            ax.errorbar(x_data, y_data_male, xerr=xerrors, fmt='x', label=f'{category.replace("_", " ").title()} Men', capsize=5,
                        color=colors[i], alpha=0.7)
            ax.errorbar(x_data, y_data_female, xerr=xerrors, fmt='^', label=f'{category.replace("_", " ").title()}: Women', capsize=5,
                        color=colors[i], alpha=0.9)

            # Plot the normalized smooth curves for both male and female
            df_male = interpolated_data.loc[(local_authority_name, 'm', category)]
            df_female = interpolated_data.loc[(local_authority_name, 'f', category)]

            x_smooth = df_male.index.values
            y_smooth_male = df_male.values.flatten()
            y_smooth_female = df_female.values.flatten()

            ax.plot(x_smooth, y_smooth_male, '--', color=colors[i], alpha=0.7)
            ax.plot(x_smooth, y_smooth_female, '-', color=colors[i], alpha=0.9)

        ax.set_title(f"Population by Partnership Status in {local_authority_name.title()} for Males and Females")
        ax.set_ylabel("Fraction of Population")
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right')

        ax.set_xlabel("Age Bracket")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xticks(age_labels, [label for label in age_labels], rotation=45)
        plt.tight_layout()
        plt.savefig(f'{paths.data_path}/graphs/partnerships/{local_authority_name}.png', dpi=1200, transparent=True, bbox_inches='tight')
        plt.show()

# Example usage:
file_paths = FilePathConfig()
data_processor = DataProcessor(file_paths)
# Uncomment the following line to process all local authorities and generate CSV files
# data_processor.process_all_local_authorities()

data_plotter = DataPlotter(file_paths)
local_authority_names = ['Cambridge']
for local_authority_name in local_authority_names:
    data_plotter.plot_partnership_status(local_authority_name)
