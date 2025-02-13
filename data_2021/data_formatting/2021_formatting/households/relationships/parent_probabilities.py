import numpy as np
import pandas as pd
from scipy.stats import skewnorm
from scipy.optimize import curve_fit
from scipy.signal import convolve
import matplotlib.pyplot as plt

from VirtUK import paths


class FertilityAnalysis:
    def __init__(self, female_population_fp, fert_rate_fp, geo_hierarchy_fp):
        # Load the data
        self.female_population_df = pd.read_csv(female_population_fp)
        self.fert_rate_df = pd.read_csv(fert_rate_fp)
        self.geo_hierarchy_df = pd.read_csv(geo_hierarchy_fp)

        # Set up indices for easier lookup
        self._prepare_dataframes()

    def _prepare_dataframes(self):
        """
        Prepares dataframes by setting indices for efficient lookup and sorting.
        """
        self.fert_rate_df.set_index('year', inplace=True)
        self.female_population_df.set_index('area', inplace=True)
        self.geo_hierarchy_df.set_index('msoa', inplace=True)
        self.geo_hierarchy_df.sort_index(inplace=True)

    def _get_msoa_output_areas(self, msoa):
        """
        Get the output areas corresponding to a given MSOA.
        """
        return self.geo_hierarchy_df.loc[self.geo_hierarchy_df.index == msoa, 'area'].tolist()

    def _aggregate_female_population(self, msoa_output_areas):
        """
        Aggregates the female population for the given MSOA output areas.
        """
        return self.female_population_df.loc[msoa_output_areas].sum()

    def _calculate_probabilities(self, msoa_female_population):
        """
        Calculates the unnormalized and normalized probabilities for each child age.
        """
        unnorm_probabilities = np.zeros((25, len(self.fert_rate_df.columns), 2))

        for child_age in range(0, 25):
            year_of_birth = 2021 - child_age

            if year_of_birth not in self.fert_rate_df.index:
                continue

            fertility_row = self.fert_rate_df.loc[year_of_birth]

            for idx, age_range in enumerate(fertility_row.index):
                ml, mh = map(int, age_range.split('-'))
                min_female_age = ml + child_age
                max_female_age = mh + child_age

                fertile_population = msoa_female_population[min_female_age:max_female_age + 1]
                unnorm_probability = (fertile_population.sum() * fertility_row.loc[age_range]) / 1000

                central_age = (min_female_age + max_female_age) // 2 + 0.5

                unnorm_probabilities[child_age, idx, 0] = central_age
                unnorm_probabilities[child_age, idx, 1] = unnorm_probability

        return self._normalize_probabilities(unnorm_probabilities)

    def _normalize_probabilities(self, unnorm_probabilities):
        """
        Normalizes probabilities along each child age dimension.
        """
        probabilities = np.zeros_like(unnorm_probabilities)
        for child_age in range(25):
            total_prob = unnorm_probabilities[child_age, :, 1].sum()
            if total_prob > 0:
                probabilities[child_age, :, 0] = unnorm_probabilities[child_age, :, 0]
                probabilities[child_age, :, 1] = unnorm_probabilities[child_age, :, 1] / (total_prob * 5)

        return probabilities

    def _fit_skewed_normal(self, central_ages, probs):
        """
        Fit a skewed normal distribution to the data points.
        """
        valid_indices = probs > 0
        if np.sum(valid_indices) <= 1:
            return None, None

        central_ages_valid = central_ages[valid_indices]
        probs_valid = probs[valid_indices]
        initial_guess = [4, np.mean(central_ages_valid), np.std(central_ages_valid)]

        try:
            popt, _ = curve_fit(self._skewed_normal_function, central_ages_valid, probs_valid, p0=initial_guess,
                                maxfev=10000)
            return central_ages_valid, popt
        except RuntimeError:
            return None, None

    @staticmethod
    def _skewed_normal_function(x, a, loc, scale):
        return skewnorm.pdf(x, a, loc, scale)

    def _save_discrete_fit_values(self, msoa, child_age, central_age_fit, probs_fit):
        """
        Saves the discrete values of the fitted probabilities for later interpolation and plotting.
        """
        if not hasattr(self, 'fitted_values_dict'):
            self.fitted_values_dict = {}

        if msoa not in self.fitted_values_dict:
            self.fitted_values_dict[msoa] = {}

        if child_age not in self.fitted_values_dict[msoa]:
            self.fitted_values_dict[msoa][child_age] = []

        self.fitted_values_dict[msoa][child_age].append((central_age_fit, probs_fit))

    def save_all_fit_values(self, output_file):
        """
        Saves all the fitted values into a numpy file for later use.
        """
        np.save(output_file, self.fitted_values_dict)

    def process_msoa(self, msoa):
        """
        Processes a specific MSOA and saves the corresponding probabilities.
        """
        msoa_output_areas = self._get_msoa_output_areas(msoa)
        if not msoa_output_areas:
            print(f"No output areas found for MSOA: {msoa}")
            return

        msoa_female_population = self._aggregate_female_population(msoa_output_areas)
        probabilities = self._calculate_probabilities(msoa_female_population)

        for child_age in range(0, 25):
            central_ages = probabilities[child_age, :, 0]
            probs = probabilities[child_age, :, 1]

            central_ages_valid, popt = self._fit_skewed_normal(central_ages, probs)
            if popt is not None:
                central_age_fit = np.linspace(min(central_ages_valid) - 5, max(central_ages_valid) + 5, 200)
                probs_fit = self._skewed_normal_function(central_age_fit, *popt)

                # Ensure step down to 0 at the lower bound
                probs_fit[central_age_fit < min(central_ages_valid) - 2.5] = 0
                probs_fit[central_age_fit > max(central_ages_valid) + 2.5] = 0

                dx = central_age_fit[1] - central_age_fit[0]
                area_under_curve = np.sum(probs_fit) * dx
                if area_under_curve > 0:
                    probs_fit = probs_fit / area_under_curve

                # Save the discrete values of the fitted probabilities for later use
                self._save_discrete_fit_values(msoa, child_age, central_age_fit, probs_fit)

        self.original_probabilities = probabilities


class FertilityPlotter:
    def __init__(self, fitted_values_fp, original_probabilities):
        """
        Initializes the plotter with the fitted values file and original probabilities.
        """
        self.fitted_values_dict = np.load(fitted_values_fp, allow_pickle=True).item()
        self.original_probabilities = original_probabilities

    def plot_probabilities(self, msoa, child_ages_to_plot):
        """
        Plots the probabilities for specified child ages for a given MSOA.
        """
        if msoa not in self.fitted_values_dict:
            print(f"No data found for MSOA: {msoa}")
            return

        plt.figure(figsize=(10, 6))
        colors = plt.cm.plasma(np.linspace(0, 1, len(child_ages_to_plot)))

        # Plot the original probabilities with error bars
        for i, child_age in enumerate(child_ages_to_plot):
            child_age_data = self.original_probabilities[child_age, :, :]
            for central_age, prob in child_age_data:
                plt.errorbar(central_age, prob, xerr=2.5, fmt='x', color=colors[i], alpha=0.5)

        for i, child_age in enumerate(child_ages_to_plot):
            if child_age in self.fitted_values_dict[msoa]:
                central_age_fit, probs_fit = self.fitted_values_dict[msoa][child_age][0]
                plt.plot(central_age_fit, probs_fit, linestyle='-', color=colors[i], label=f'Child Age {child_age}')

        plt.xlabel('Mother Age (Central Point of Age Range)')
        plt.ylabel('Normalized Probability')
        plt.title('Probability vs. Mother Age for Different Child Ages with Skewed Normal Fit')
        plt.grid(False)
        plt.xlim(14, 70)
        plt.ylim(0)
        plt.show()

    def plot_heatmap(self, msoa):
        """
        Plots a heatmap of probabilities for child ages 0 to 24 (rows) and parent ages 16 to 70 (columns),
        including lines for parent age = child age + 20, +25, +30, +35, +40.
        """
        # Define the age ranges
        child_ages = np.arange(0, 25)
        parent_ages = np.arange(16, 71)  # 16 to 70 inclusive

        # Initialize the 2D array for the heatmap
        heatmap_data = np.zeros((len(child_ages), len(parent_ages)))

        # Check if data exists for the given MSOA
        if msoa not in self.fitted_values_dict:
            print(f"No data found for MSOA: {msoa}")
            return

        for i, child_age in enumerate(child_ages):
            # Each child_age entry should have one tuple: (central_age_fit, probs_fit)
            if child_age not in self.fitted_values_dict[msoa]:
                # No fitted data for this child_age, remain zeros
                continue

            central_age_fit, probs_fit = self.fitted_values_dict[msoa][child_age][0]

            # Interpolate probabilities at each integer parent age using np.interp
            if len(central_age_fit) == 0 or len(probs_fit) == 0:
                continue

            interpolated_probs = np.interp(parent_ages, central_age_fit, probs_fit, left=0, right=0)
            heatmap_data[i, :] = interpolated_probs

        # Plot the heatmap
        plt.figure(figsize=(10, 6))
        im = plt.imshow(heatmap_data, aspect='auto', origin='lower',
                        extent=[parent_ages[0], parent_ages[-1], child_ages[0], child_ages[-1]],
                        cmap='plasma')
        plt.colorbar(im, label='Probability')

        # Set ticks at 16 and every multiple of 5 after that
        parent_tick_locs = [16] + list(np.arange(20, 71, 5))
        plt.xticks(parent_tick_locs)
        child_tick_locs = np.arange(0, 25, 5)
        plt.yticks(child_tick_locs)

        # Add the lines for parent age = child age + 20, +25, +30, +35, +40
        offsets = [20, 25, 30, 35, 40]
        styles = {
            20: (0, (1, 1)),  # Dotted line
            25: (0, (5, 5)),  # Dashed line
            30: '-',  # Solid line
            35: (0, (5, 5)),  # Dashed line
            40: (0, (1, 1))  # Dotted line
        }
        colors = 'white'

        for offset in offsets:
            x_line = np.arange(16, 71)  # Parent age range
            y_line = x_line - offset  # Corresponding child age
            valid_indices = (y_line >= 0) & (y_line <= 24)  # Keep within bounds of child age
            plt.plot(x_line[valid_indices], y_line[valid_indices], linestyle=styles[offset], color=colors, linewidth=2,
                     label=f'Parent Age = Child Age + {offset}')

        # Add labels, legend, and title
        plt.xlabel("Mother's Age")
        plt.ylabel('Child Age')
        plt.title("Heatmap of Probability Distributions (Child Age vs Mother's Age)")
        plt.legend()

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14

        plt.savefig('parent_probabilities_heatmap', dpi=1200, transparent=True, bbox_inches='tight')
        plt.show()


# Example usage:
if __name__ == "__main__":
    female_population_fp = f'{paths.data_path}/input/demography/age_dist_2021/1_year_ages_female_oa.csv'
    fert_rate_fp = f'{paths.data_path}/input/households/age_fertility_year.csv'
    geo_hierarchy_fp = f'{paths.data_path}/input/geography/oa_msoa_lad_regions.csv'
    output_distributions_fp = f'{paths.data_path}/input/households/fitted_child_parent.npy'

    analysis = FertilityAnalysis(female_population_fp, fert_rate_fp, geo_hierarchy_fp)
    # msoa = 'E02006527'
    msoa = 'E02004315'# For Durham
    analysis.process_msoa(msoa)
    analysis.save_all_fit_values(output_distributions_fp)

    plotter = FertilityPlotter(output_distributions_fp, analysis.original_probabilities)
    child_ages_to_plot = np.arange(0, 25, 1)
    plotter.plot_probabilities(msoa, child_ages_to_plot)
    plotter.plot_heatmap(msoa)
