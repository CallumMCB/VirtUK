import pandas as pd
from tabulate import tabulate

from VirtUK import paths

companies_path = f"{paths.data_path}/raw_data/companies/census2021-ts060-msoa.csv"

# Read the CSV file
df = pd.read_csv(companies_path)

# Select columns that match the criteria
filtered_columns = ['geography code'] + [
    col for col in df.columns if col.startswith('Industry (current):') and
    col[len('Industry (current): ')].isalpha() and
    col[len('Industry (current): ')].isupper()
]

# Rename the columns by removing the 'Industry (current): ' prefix
df_filtered = df[filtered_columns].rename(
    columns=lambda x: x.replace('Industry (current): ', '') if x.startswith('Industry (current): ') else x
)

# Save the filtered dataframe to a new CSV file
filtered_file_path = f"{paths.data_path}/input/companies/companies_employees.csv"
df_filtered.to_csv(filtered_file_path, index=False)


