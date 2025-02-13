import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from VirtUK import paths
import os

# File paths and initial setup
year = 2021
age_data_dir = f'{paths.data_path}/input/demography/age_dist_{year}/'
file_types = ['m', 'f']
geography_dir = f'{paths.data_path}/input/geography/'
hierarchy_fp = geography_dir + 'oa_msoa_lad_regions.csv'

# Load the hierarchy data
hierarchy = pd.read_csv(hierarchy_fp)


# Function to process data for exponential fitting
def process_data_for_exponential_fitting(file_type, age_data_dir, hierarchy):
    """
    Process data to calculate exponential fitting parameters for each LAD.

    Parameters
    ----------
    file_type : str
        Type of file ('m' or 'f').
    age_data_dir : str
        Directory containing age data files.
    hierarchy : pd.DataFrame
        DataFrame containing area to LAD mapping.

    Returns
    -------
    dict
        A dictionary with LAD as keys and (intercept, decay_rate) as values.
    """
    # Load age data for output areas
    fp = f'{age_data_dir}1_year_ages_{file_type}_oa.csv'
    df = pd.read_csv(fp)

    # Merge with hierarchy to get LAD information
    df = df.merge(hierarchy[['area', 'lad_code']], left_on='area', right_on='area', how='left')

    # Group by LAD and average the age distributions
    grouped = df.groupby('lad_code').mean()

    # Fit exponential decay for ages 84-89
    age_columns = [str(age) for age in range(84, 90)]
    results = {}
    for lad, row in grouped.iterrows():
        counts_84_89 = row[age_columns].values
        ages_84_89 = np.arange(84, 90)
        counts_84_89 = np.where(counts_84_89 == 0, 0.0001, counts_84_89)  # Avoid zeros for fitting
        log_counts = np.log(counts_84_89)

        # Fit exponential decay
        decay_rate, intercept = np.polyfit(ages_84_89, log_counts, 1)
        results[lad] = (intercept, decay_rate)

    return results


# Function to apply exponential decay to age counts
def apply_exponential_decay(age_counts, count_value, intercept, decay_rate, ages_90_100_range=(90, 101), num_output_areas=1):
    """
    Applies an exponential decay based on the provided intercept and decay rate until no counts remain.

    Parameters
    ----------
    age_counts : np.array
        Array of age counts to be updated.
    count_value : int
        Number of counts to distribute according to the exponential decay.
    intercept : float
        Intercept value for the exponential decay.
    decay_rate : float
        Decay rate for the exponential decay.
    ages_90_100_range : tuple
        Age range to apply the decay, default is (90, 101).

    Returns
    -------
    np.array
        Updated age counts including exponential decay applied for the given range.
    """

    ages = np.arange(*ages_90_100_range)
    exp_values = np.exp(intercept + decay_rate * ages)*num_output_areas

    # Prevent negative or very small values
    exp_values = np.maximum(exp_values, 0.0001)

    # Normalize exponential values to make sure they sum up to the count_value
    total_exp_sum = np.sum(exp_values)
    if total_exp_sum == 0:
        return age_counts

    # Fill each age group successively with its predicted number of people until no counts remain
    for age, predicted_count in zip(ages, exp_values):
        predicted_count = int(predicted_count)
        if predicted_count > count_value:
            predicted_count = count_value
        age_counts[age] += predicted_count
        count_value -= predicted_count
        if count_value <= 0:
            break

    return age_counts


# Function to sample age distribution for specific LADs
def sample_age_distribution(lad, male_df, female_df, hierarchy, intercept_decay_results):
    """
    Sample age distribution for a specific LAD.

    Parameters
    ----------
    lad : str
        LAD to sample.
    male_df : pd.DataFrame
        DataFrame containing male age data.
    female_df : pd.DataFrame
        DataFrame containing female age data.
    hierarchy : pd.DataFrame
        DataFrame containing area to LAD mapping.
    intercept_decay_results : dict
        Dictionary containing intercept and decay rates for each LAD.

    Returns
    -------
    tuple
        A tuple of male and female age counts.
    """
    if lad not in intercept_decay_results['m'] or lad not in intercept_decay_results['f']:
        print(lad)
        return np.zeros(101), np.zeros(101)  # Return zero counts if LAD is not found in the intercept results

    intercept_male, decay_rate_male = intercept_decay_results['m'][lad]
    intercept_female, decay_rate_female = intercept_decay_results['f'][lad]

    male_age_counts = np.zeros(101)
    female_age_counts = np.zeros(101)

    # Use the existing counts for ages below 90
    male_age_counts[:90] = male_df[male_df['lad_code'] == lad].iloc[:, 1:91].sum().values
    female_age_counts[:90] = female_df[female_df['lad_code'] == lad].iloc[:, 1:91].sum().values

    # Apply exponential decay for ages 90+
    count_value_male = male_df[male_df['lad_code'] == lad].iloc[:, 91].sum()
    num_output_areas = len(hierarchy[hierarchy['lad_code'] == lad])
    male_age_counts = apply_exponential_decay(male_age_counts, count_value=int(count_value_male),
                                              intercept=intercept_male,
                                              decay_rate=decay_rate_male, ages_90_100_range=(90, 101),
                                              num_output_areas=num_output_areas)
    count_value_female = female_df[female_df['lad_code'] == lad].iloc[:, 91].sum()
    female_age_counts = apply_exponential_decay(female_age_counts, count_value=int(count_value_female),
                                                intercept=intercept_female,
                                                decay_rate=decay_rate_female, ages_90_100_range=(90, 101),
                                                num_output_areas=num_output_areas)
    return male_age_counts, female_age_counts


# Function to plot population pyramid
def plot_population_pyramid(lad, male_age_counts, female_age_counts, hierarchy, intercept_decay_results):
    """
    Plot the population pyramid for a specific LAD.

    Parameters
    ----------
    lad : str
        LAD to plot.
    male_age_counts : np.array
        Array of male age counts.
    female_age_counts : np.array
        Array of female age counts.
    hierarchy : pd.DataFrame
        DataFrame containing area to LAD mapping.
    intercept_decay_results : dict
        Dictionary containing intercept and decay rates for each LAD.
    """
    intercept_male, decay_rate_male = intercept_decay_results['m'][lad]
    intercept_female, decay_rate_female = intercept_decay_results['f'][lad]

    # Plot the population pyramid
    ages = np.arange(101)
    fig, ax = plt.subplots(figsize=(10, 8))
    male_bars = ax.barh(ages, male_age_counts, color='lightblue', label='Males')
    female_bars = ax.barh(ages, -female_age_counts, color='lightcoral', label='Females')

    # Highlight excess counts in darker colors
    for male_bar, female_bar in zip(male_bars, female_bars):
        male_height = male_bar.get_width()
        female_height = -female_bar.get_width()

        if male_height > female_height:
            excess_width = male_height - female_height
            ax.barh(male_bar.get_y(), excess_width, height=male_bar.get_height(), color='darkblue', left=female_height)
        elif female_height > male_height:
            excess_width = female_height - male_height
            ax.barh(female_bar.get_y(), -excess_width, height=female_bar.get_height(), color='darkred',
                    left=-male_height)

    # Plot the fitted exponential curves
    ages_84_101 = np.arange(84, 101)
    num_output_areas = len(hierarchy[hierarchy['lad_code'] == lad])
    male_fit_values = np.exp(intercept_male + decay_rate_male * ages_84_101) * num_output_areas
    female_fit_values = np.exp(intercept_female + decay_rate_female * ages_84_101) * num_output_areas
    ax.plot(male_fit_values, ages_84_101, color='blue', linestyle='--', label='Male Fit (84-101)')
    ax.plot(-female_fit_values, ages_84_101, color='red', linestyle='--', label='Female Fit (84-101)')

    # Set symmetrical x-axis limits
    max_population = max(male_age_counts.max(), female_age_counts.max())
    ax.set_xlim(-max_population * 1.1, max_population * 1.1)

    # Set y-axis limits and add dotted lines every 5 years
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 5))
    ax.grid(axis='y', linestyle=':', color='grey')

    ax.set_xlabel('Population')
    ax.set_ylabel('Age')
    ax.set_title(f'Population Pyramid for LAD {lad} - Year {year}')
    ax.legend()
    plt.show()


# Main script execution
if __name__ == "__main__":
    # Process data for both 'm' and 'f'
    intercept_decay_results = {}
    for file_type in file_types:
        intercept_decay_results[file_type] = process_data_for_exponential_fitting(file_type, age_data_dir, hierarchy)

    # Load the initial age data
    male_age_fp = f'{age_data_dir}1_year_ages_male_oa.csv'
    female_age_fp = f'{age_data_dir}1_year_ages_female_oa.csv'
    male_df = pd.read_csv(male_age_fp)
    female_df = pd.read_csv(female_age_fp)

    # Merge with hierarchy to get LAD information
    male_df = male_df.merge(hierarchy[['area', 'lad_code']], left_on='area', right_on='area', how='left')
    female_df = female_df.merge(hierarchy[['area', 'lad_code']], left_on='area', right_on='area', how='left')

    sampled_populations = {}
    for lad in hierarchy['lad_code'].unique():
        male_age_counts, female_age_counts = sample_age_distribution(lad, male_df, female_df, hierarchy,
                                                                     intercept_decay_results)
        sampled_populations[lad] = (male_age_counts, female_age_counts)

    exp_fit_df = pd.DataFrame([(lad, file_type, intercept, decay_rate)
                               for file_type, data in intercept_decay_results.items()
                               for lad, (intercept, decay_rate) in data.items()],
                              columns=['lad_code', 'Gender', 'Intercept', 'DecayRate'])
    exp_fit_df.to_csv(os.path.join(age_data_dir, 'exponential_curves.csv'), index=False)

    # Save results to CSV files
    male_lad_df = pd.DataFrame({lad: pop[0] for lad, pop in sampled_populations.items()}).T
    male_lad_df.columns = [f'age_{i}' for i in range(101)]
    male_lad_df.index.name = 'lad_code'
    male_lad_df.sort_index(inplace=True)
    male_lad_df.to_csv(f'{age_data_dir}sampled_ages_male_lad.csv')

    female_lad_df = pd.DataFrame({lad: pop[1] for lad, pop in sampled_populations.items()}).T
    female_lad_df.columns = [f'age_{i}' for i in range(101)]
    female_lad_df.index.name = 'lad_code'
    female_lad_df.sort_index(inplace=True)
    female_lad_df.to_csv(f'{age_data_dir}sampled_ages_female_lad.csv')