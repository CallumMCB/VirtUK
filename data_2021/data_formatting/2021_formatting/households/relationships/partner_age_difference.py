import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from VirtUK import paths
from tqdm import tqdm
import os
import pickle
from scipy.interpolate import make_interp_spline
import seaborn as sns
from matplotlib.widgets import Cursor
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from matplotlib.colors import PowerNorm


@dataclass
class FilePathConfig:
    age_diff_file: str = f'{paths.data_path}/input/households/age_difference_by_female_age.csv'
    interpolation_output_file: str = f'{paths.data_path}/input/households/age_difference_interpolations.csv'
    female_age_file: str = f'{paths.data_path}/input/demography/age_Dist_2021/1_year_ages_female_oa.csv'


class AgeDifferenceProcessor:
    def __init__(self, file_paths: FilePathConfig):
        self.file_paths = file_paths
        self.age_diff_df = pd.read_csv(self.file_paths.age_diff_file)
        self.age_diff_df.columns = self.age_diff_df.columns.astype(str)  # Ensure columns are strings for consistency
        self.processed_data = pd.DataFrame()
        self.interpolated_data = pd.DataFrame()
        self.female_age_df = pd.read_csv(self.file_paths.female_age_file).drop(columns=['all_ages'])

    def process_age_difference_data(self):
        # We do not need hierarchical separation as in the previous code, so we'll work directly with the provided data
        self.age_diff_df.set_index('male relative age', inplace=True)
        # No additional processing is required as we just need to visualize the data directly
        self.processed_data = self.age_diff_df

        print("Age difference data processing complete.")

    def calculate_median_ages(self):
        median_ages = []
        ranges = []
        for col in self.age_diff_df.columns:
            age_low, age_high = col.split('-')
            age_low, age_high = int(age_low), int(age_high)
            ranges.append(age_high - age_low)

            age_distribution = self.female_age_df.iloc[:, age_low:age_high + 1].sum(axis=0)
            total_population = age_distribution.sum()
            cumulative_sum = age_distribution.cumsum()
            median_age_index = cumulative_sum.searchsorted(total_population / 2)
            median_age = age_distribution.index[median_age_index]  # Median age within the bracket
            median_ages.append(float(median_age))

        return median_ages, ranges

    @staticmethod
    def transform_dataframe(df):
        # Create an empty DataFrame for the transformed data
        new_data = pd.DataFrame(0, index=range(-20, 21), columns=df.columns)

        # Iterate through each row and column to calculate the new index
        for row_index, row in df.iterrows():
            if -20 <= int(row_index) <= 20:
                for col_index in df.columns:
                    # Only proceed if the column index is within [-20, 20]

                    # Calculate the new row index by adding the row index and column index
                    new_index = row_index + int(col_index)

                    # Only add values within the valid age range [15, 99]
                    if 16 <= new_index:
                        if new_index > 99:
                            weight = 0.5 ** (new_index - 99)
                        else:
                            weight = 1
                        new_data.at[row_index, col_index] += row[col_index] * weight
                        # new_data.at[new_index, col_index] += row[col_index]


        # Normalize rows to sum to 1
        col_sums = new_data.sum(axis=0)
        print(col_sums)
        new_data = new_data.div(col_sums, axis=1)

        return new_data

    def perform_interpolation_and_store(self):
        # Interpolate and store the age difference data
        median_ages, ranges = self.calculate_median_ages()
        x_numeric = np.array(median_ages)
        x_smooth = np.arange(16, 101, 1)

        interpolated_category_data = []

        for age_diff in self.age_diff_df.index:
            y_data = self.age_diff_df.loc[age_diff].values
            y_data = np.clip(y_data, 0, None)  # Clip values below 0 to 0

            # Create a spline interpolation for smoother lines
            try:
                spline = make_interp_spline(x_numeric, y_data, k=3)
                y_smooth = spline(x_smooth)
                y_smooth = np.clip(y_smooth, 0, None)  # Clip values below 0 to 0
            except ValueError as e:
                print(f"Skipping age_diff {age_diff} due to interpolation error: {e}")
                continue

            interpolated_category_data.append((age_diff, y_smooth))

        # Normalize such that at every x position, the sum of all lines equals 1
        interpolated_data_matrix = np.array([y_smooth for _, y_smooth in interpolated_category_data])
        total_sum_at_each_x = interpolated_data_matrix.sum(axis=0, where=(interpolated_data_matrix > 0))
        total_sum_at_each_x[total_sum_at_each_x == 0] = 1  # Prevent division by zero

        normalized_interpolated_data_matrix = interpolated_data_matrix / total_sum_at_each_x

        # Store the normalized results in a DataFrame with MultiIndex
        for i, (age_diff, _) in enumerate(interpolated_category_data):
            y_smooth_normalized = normalized_interpolated_data_matrix[i]
            df = pd.DataFrame(y_smooth_normalized.reshape(1, -1), columns=x_smooth, index=[age_diff])
            self.interpolated_data = pd.concat([self.interpolated_data, df], axis=0)

        # Transform and save the data
        final_data = self.transform_dataframe(self.interpolated_data)
        print(final_data)

        # Plot the heatmap
        AgeDifferencePlotter.plot_heatmap(
            final_data.loc[-20:20, final_data.columns.intersection(range(16, 101))],
            title="Heatmap Normalized by Female Age (Ages 16-99)"
        )

        # Save the interpolated data to a CSV file
        final_data.to_csv(self.file_paths.interpolation_output_file)

        # # Normalize in the other direction and plot
        # final_data_normalized_by_male_age = final_data.div(final_data.sum(axis=1), axis=0)
        # AgeDifferencePlotter.plot_heatmap(
        #     final_data_normalized_by_male_age.loc[-20:20,
        #     final_data_normalized_by_male_age.columns.intersection(range(16, 101))],
        #     title="Heatmap Normalized by Male Age (Ages 16-99)"
        # )

        print("Interpolation complete and data stored.")


class AgeDifferencePlotter:
    def __init__(self, age_diff_df, median_ages, ranges):
        self.age_diff_df = age_diff_df
        self.median_ages = median_ages
        self.ranges = ranges
        self.age_differences = self.age_diff_df.index.values
        self.age_diff_values = self.age_diff_df.values  # Values corresponding to each age bin and age difference

    @staticmethod
    def plot_heatmap(df, title='85x85 Grid Heatmap of Values'):
        # Ensure the DataFrame index and columns include the full range
        missing_indices = [idx for idx in range(-20, 21) if idx not in df.index]
        missing_columns = [col for col in range(16, 101) if col not in df.columns]

        if missing_indices:
            print(f"Warning: The DataFrame is missing the following indices: {missing_indices}")
        if missing_columns:
            print(f"Warning: The DataFrame is missing the following columns: {missing_columns}")

        # Fill missing indices and columns with zeros
        df = df.reindex(index=range(-20, 21), columns=range(16, 101), fill_value=0)

        # Create a custom colormap that sets zero values to white
        cmap = plt.cm.jet  # Change colormap as needed
        cmap.set_under(color='white')

        fig, ax = plt.subplots(figsize=(15, 3))
        heatmap = sns.heatmap(
            df.loc[-20:21, range(16, 101)],  # Ensure full range is included
            annot=False,
            fmt=".4f",
            cmap=cmap,
            cbar=True,
            linewidths=.5,
            ax=ax,
            norm=PowerNorm(gamma=0.5),
            vmin=0.000001,  # Set minimum value for colormap
            linecolor='black'
        )

        plt.title(title)
        plt.ylabel('Male Relative Age')

        # Plotting lines across the heatmap
        x = np.arange(0, 83)
        ax.plot(x, [35.5]*len(x), color='white', linestyle=':', linewidth=2, label='+15')  # Diagonal line y=15
        ax.plot(x, [30.5]*len(x), color='white', linestyle='-.', linewidth=2, label='+10')  # Diagonal line y=10
        ax.plot(x, [25.5]*len(x), color='white', linestyle='--', linewidth=2, label='+5')  # Diagonal line y=10
        ax.plot(x, [20.5]*len(x), color='white', linestyle='-', linewidth=2, label='Same Age')  # Diagonal line y=0
        ax.plot(x, [15.5]*len(x), color='white', linestyle='--', linewidth=2, label='-10')  # Diagonal line y=-5
        ax.plot(x, [10.5]*len(x), color='white', linestyle='-.', linewidth=2, label='-10')  # Diagonal line y=-10
        ax.plot(x, [5.5]*len(x), color='white', linestyle=':', linewidth=2, label='-15')  # Diagonal line y=-15

        # Set limits for the grid
        ax.set_xlim(0, 82)
        ax.set_ylim(0, 41)

        y_ticks = np.arange(0, 41, 1)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(tick -20) if tick % 5 == 0 else '' for tick in y_ticks])
        x_ticks = np.arange(0,82,1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(tick+16) if ((tick+1) % 5 == 0 or tick == 0) else '' for tick in x_ticks])
        ax.set_xlabel('Female Age')

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14

        fig.savefig("relative_partner_age.png", dpi=1200, transparent=True)

        # Function to handle click events
        def on_click(event):
            if event.inaxes == ax:
                x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)  # Round to nearest integer
                if x in df.columns and y in df.index:
                    value = df.at[y, x]
                    print(f"Probability at (Age={x}, Transformed Male Age={y}): {value:.4f}")
                    text_color = 'white' if value != 0 else 'black'
                    ax.annotate(
                        f"{value:.4f}",
                        (x, y),
                        color=text_color,
                        ha='center',
                        va='center',
                        fontsize=8,
                        weight='bold'
                    )
                    fig.canvas.draw()

        # Connect the click event to the function
        fig.canvas.mpl_connect('button_press_event', on_click)

        # Set gridlines color to black
        ax.grid(color='black', linestyle='-', linewidth=0.5)

        # Convert the heatmap to Plotly and save as an HTML file for interactive use
        plotly_fig = go.Figure()
        plotly_fig.add_trace(
            go.Heatmap(
                z=df.values,
                x=df.columns,
                y=df.index,
                colorscale='jet',
                zmin=0.000001,
                hoverongaps=False
            )
        )

        # Add diagonal lines similar to the original Matplotlib version
        plotly_fig.add_trace(
            go.Scatter(x=x, y=x - 20, mode='lines', line=dict(color='white', width=2, dash='solid'), hoverinfo='skip'))
        plotly_fig.add_trace(
            go.Scatter(x=x, y=x, mode='lines', line=dict(color='white', width=2, dash='dash'), hoverinfo='skip'))
        plotly_fig.add_trace(
            go.Scatter(x=x, y=x + 20, mode='lines', line=dict(color='white', width=2, dash='dot'), hoverinfo='skip'))

        plotly_fig.update_layout(
            title=title,
            xaxis_title='Female Age',
            yaxis_title='Male Age',
            xaxis=dict(showgrid=True, gridcolor='black'),
            yaxis=dict(showgrid=True, gridcolor='black'),
            plot_bgcolor='white'
        )
        plotly_fig.write_html("heatmap.html")
        print("Interactive heatmap saved as 'heatmap.html'.")

    def plot_age_difference_bars(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        bar_positions = []
        bar_widths = []
        for col in self.age_diff_df.columns:
            age_low, age_high = col.split('-')
            age_low, age_high = int(age_low), int(age_high)
            bar_positions.append((age_low + age_high) / 2)
            bar_widths.append(age_high - age_low + 1)

        x_positions = np.array(bar_positions)

        # Use distinct color maps for negative and positive values without fading to white
        cmap_negative = plt.get_cmap('RdYlGn_r')  # Using autumn for negative values (distinct reddish-orange hues)
        cmap_positive = plt.get_cmap('PuBuGn_r')  # Reversed winter for positive values (distinct blue-green hues)
        norm_negative = plt.Normalize(vmin=-25, vmax=-1)
        norm_positive = plt.Normalize(vmin=1, vmax=25)

        bottom = np.zeros(len(self.age_diff_df.columns))
        print(x_positions, bar_widths)
        print(self.age_differences)
        for i, age_diff in enumerate(self.age_differences):
            if age_diff < 0:
                color = cmap_negative(norm_negative(age_diff))
            elif age_diff > 0:
                color = cmap_positive(norm_positive(age_diff))
            else:
                color = 'black'  # Use black for 0 age difference
            ax.bar(x_positions, self.age_diff_values[i], width=bar_widths, bottom=bottom, color=color, alpha=0.7)
            bottom += self.age_diff_values[i]

        ax.set_xlabel("Female Age Bracket")
        ax.set_ylabel("Fraction of Population")
        ax.set_title("Age Difference Distribution by Female Age Bracket")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'{int(pos - width/2)}-{int(pos + width/2)}' for pos, width in zip(bar_positions, bar_widths)], rotation=45)

        # Creating two color bars to indicate the heatmap effect
        sm_neg = plt.cm.ScalarMappable(cmap=cmap_negative, norm=norm_negative)
        sm_neg.set_array([])
        cbar_neg = plt.colorbar(sm_neg, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar_neg.set_label('Male Relative Age Difference (Negative)')

        sm_pos = plt.cm.ScalarMappable(cmap=cmap_positive, norm=norm_positive)
        sm_pos.set_array([])
        cbar_pos = plt.colorbar(sm_pos, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar_pos.set_label('Male Relative Age Difference (Positive)')

        plt.tight_layout()
        plt.show()

    def plot_interpolated_lines(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Use distinct color maps for negative and positive values without fading to white
        cmap_negative = plt.get_cmap('hot_r')  # Using autumn for negative values (distinct reddish-orange hues)
        cmap_positive = plt.get_cmap('cool')  # Reversed winter for positive values (distinct blue-green hues)
        norm_negative = plt.Normalize(vmin=-25, vmax=-1)
        norm_positive = plt.Normalize(vmin=1, vmax=25)

        x_numeric = np.array(self.median_ages)
        x_smooth = np.linspace(16, 100, 300)
        for i, age_diff in enumerate(self.age_differences):
            y_data = self.age_diff_values[i]
            y_data = np.clip(y_data, 0, None)  # Clip values below 0 to 0

            # Plot the original data points for reference
            if age_diff < 0:
                color = cmap_negative(norm_negative(age_diff))
            elif age_diff > 0:
                color = cmap_positive(norm_positive(age_diff))
            else:
                color = 'black'  # Use black for 0 age difference
            ax.plot(x_numeric, y_data, 'o', label=f'Original {age_diff}', color=color, alpha=0.5, markersize=4)

            # Create a spline interpolation for smoother lines
            spline = make_interp_spline(x_numeric, y_data, k=3)
            y_smooth = spline(x_smooth)

            # Clip values below 0 and normalize across all age differences to ensure sum is 1 at each x value
            y_smooth = np.clip(y_smooth, 0, None)

            ax.plot(x_smooth, y_smooth, label=f'Interpolated {age_diff}', color=color, alpha=0.8)

        ax.set_xlabel("Median Female Age")
        ax.set_ylabel("Fraction of Population")
        ax.set_title("Smoothed Age Difference Distribution by Female Age")
        ax.set_xticks(x_numeric)
        ax.set_xticklabels([f'{age:.1f}' for age in self.median_ages], rotation=45)

        # Creating two color bars to indicate the heatmap effect
        sm_neg = plt.cm.ScalarMappable(cmap=cmap_negative, norm=norm_negative)
        sm_neg.set_array([])
        cbar_neg = plt.colorbar(sm_neg, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar_neg.set_label('Male Relative Age Difference (Negative)')

        sm_pos = plt.cm.ScalarMappable(cmap=cmap_positive, norm=norm_positive)
        sm_pos.set_array([])
        cbar_pos = plt.colorbar(sm_pos, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar_pos.set_label('Male Relative Age Difference (Positive)')
        plt.tight_layout()
        plt.show()

# Example usage:
file_paths = FilePathConfig()
age_diff_processor = AgeDifferenceProcessor(file_paths)
age_diff_processor.process_age_difference_data()
age_diff_processor.perform_interpolation_and_store()

# Instantiate the AgeDifferencePlotter with the processed DataFrame
median_ages, ranges = age_diff_processor.calculate_median_ages()
age_difference_plotter = AgeDifferencePlotter(age_diff_processor.processed_data, median_ages, ranges)

# Plot bar chart with age differences
age_difference_plotter.plot_age_difference_bars()

# Plot interpolated lines for the smoothed age differences
age_difference_plotter.plot_interpolated_lines()
