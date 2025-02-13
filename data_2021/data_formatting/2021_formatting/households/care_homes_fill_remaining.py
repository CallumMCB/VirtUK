import pandas as pd
import numpy as np
from tabulate import tabulate
from VirtUK import paths

# Path to the CSV output from your previous processing
dir = f'{paths.data_path}/input/households/communal_establishments/care_homes/'
input_file = f'{dir}occupancy.csv'
output_file = f'{dir}occupancy_filled.csv'

# Load the data
df = pd.read_csv(input_file)

# It is assumed that the following columns exist:
#   'Location Region'     -- region for the care home
#   'Care homes beds'     -- number of beds (should be numeric)
#   '%occupancy_mean'     -- percentage occupancy (between 0 and 100), possibly with missing values

# Ensure that the bed counts are numeric
df['Care homes beds'] = pd.to_numeric(df['Care homes beds'], errors='coerce')


def random_value_from_pdf(values, bins=20):
    """
    Given an array of occupancy values (between 0 and 100),
    bin the data, interpolate the estimated density,
    and then return a random value drawn from this estimated distribution.
    """
    # Create bins spanning the entire possible range
    hist, bin_edges = np.histogram(values, bins=bins, range=(0, 100), density=False)

    total = hist.sum()
    if total == 0:
        # Log a warning and return a uniform sample
        print("Warning: No valid occupancy values found in range 0-100. Returning a uniform random value.")
        return np.random.uniform(0, 100)

    # Convert bin counts into probabilities
    bin_probs = hist / total

    # Randomly select a bin based on bin probabilities
    chosen_bin_index = np.random.choice(np.arange(len(hist)), p=bin_probs)

    # Determine the edges of the selected bin
    bin_lower = bin_edges[chosen_bin_index]
    bin_upper = bin_edges[chosen_bin_index + 1]

    # Uniformly sample within the selected bin
    random_value = np.random.uniform(bin_lower, bin_upper)

    return random_value


# List to collect details for printing a summary table of filled values
fill_records = []

total_rows = len(df)
for idx, row in df.iterrows():
    # Progress tracker: print progress every 1000 rows
    if idx % 1000 == 0:
        print(f"Processing row {idx} of {total_rows}...")

    # Skip rows that already have a value for '%occupancy_mean'
    if not pd.isnull(row['%occupancy_mean']):
        continue

    bed_range = 0.2
    max_range = 1.0
    worked = False

    # Store these outside the loop so they are available for fallback if needed.
    region = row['Location Region']
    beds = row['Care homes beds']

    if pd.isnull(beds):
        # If bed count is missing, skip this row
        continue

    while not worked and bed_range <= max_range:
        # Define ±bed_range for number of beds
        lower_beds = beds * (1 - bed_range)
        upper_beds = beds * (1 + bed_range)

        # Filter for similar care homes: same region and bed counts within range,
        # and with non-null %occupancy_mean
        similar = df[
            (df['Location Region'] == region) &
            (df['Care homes beds'] >= lower_beds) &
            (df['Care homes beds'] <= upper_beds) &
            (df['%occupancy_mean'].notnull())
        ]

        occupancy_values = similar['%occupancy_mean'].values
        occupancy_values = occupancy_values[~np.isnan(occupancy_values)]
        if len(occupancy_values) == 0:
            bed_range += 0.2
            continue

        # Get a filled occupancy value using the estimated distribution
        filled_value = random_value_from_pdf(occupancy_values, bins=20)
        df.at[idx, '%occupancy_mean'] = filled_value
        df.at[idx, 'occupancy_mean'] = round(filled_value * beds / 100, 0)

        # Also compute the histogram counts (non-normalized) for reporting.
        counts, bin_edges = np.histogram(occupancy_values, bins=20, range=(0, 100))
        # Format bins as intervals (e.g., "0-10", "10-20", …)
        bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
        # Record: Region, Beds, Filled Value, Bins, Frequency counts
        fill_records.append((region, beds, round(filled_value, 2),
                             ", ".join(bin_labels),
                             ", ".join([str(c) for c in counts])))
        worked = True

    # Fallback if we couldn't find any similar rows within the maximum range.
    if not worked:
        fallback_candidates = df[(df['Location Region'] == region) & df['%occupancy_mean'].notnull()]
        fallback_value = fallback_candidates['%occupancy_mean'].median() if not fallback_candidates.empty else np.nan
        if not pd.isnull(fallback_value):
            df.at[idx, '%occupancy_mean'] = fallback_value
            df.at[idx, 'occupancy_mean'] = round(fallback_value * beds / 100, 0)
            fill_records.append((region, beds, round(fallback_value, 2), "FALLBACK", "FALLBACK"))

print(tabulate(df, headers="keys", tablefmt="grid"))

# Save the DataFrame with filled occupancy values to a new CSV file.
df.to_csv(output_file, index=False)
print(f"Filled occupancy values saved to: {output_file}")

# Print a summary table of filled values using tabulate.
if fill_records:
    headers = ["Location Region", "Beds", "Filled %occupancy_mean", "Bins", "Frequencies"]
    print(tabulate(fill_records, headers=headers, tablefmt="grid"))
else:
    print("No missing values were filled.")
