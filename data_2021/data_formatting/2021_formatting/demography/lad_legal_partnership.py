import pandas as pd
from VirtUK import paths

def process_sheet(file_path, sheet_name, sex):
    # Load sheet into DataFrame, starting after line 5
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=5, engine='openpyxl')

    # Replace 'c' with 0
    df.replace('c', 0, inplace=True)

    # Rename columns
    df.rename(columns={
        'Area name': 'lad',
        'Area code': 'lad_code',
        'Never married and never registered a civil partnership (value)': 'never_registered_partnership',
        'Married or in a civil partnership (value)': 'married_or_civil_partnership',
        'Seperated, but still legally married or still legally in a civil partnership (value)': 'seperated',
        'Divorced or formerly in a civil partnership (value)': 'divorced',
        'Widowed or surviving partner from a civil partnership (value)': 'widowed',
        'Age of usual resident': 'age_bracket'
    }, inplace=True)

    # Check if 'separated' and 'divorced' columns exist before merging
    if 'seperated' in df.columns and 'divorced' in df.columns:
        # Merge 'separated' and 'divorced' columns into 'split'
        df['split'] = df['seperated'] + df['divorced']

        # Drop original 'separated' and 'divorced' columns
        df.drop(columns=['seperated', 'divorced'], inplace=True)
    else:
        print(df.columns)
        print(f"Warning: 'seperated' or 'divorced' column not found in sheet {sheet_name}")

    # Reformat 'Aged x to y years' to 'x-y' in 'age_bracket' column
    df['age_bracket'] = df['age_bracket'].str.replace(r'Aged (\d+) to (\d+) years', r'\1-\2', regex=True)
    df['age_bracket'] = df['age_bracket'].str.replace(r'Aged 85\+', '85+', regex=True)

    # Save the result as a CSV file
    output_dir = f'{paths.data_path}/input/demography/partnership_status/lad/'
    output_file_name = f"{sex}_legal_partnership.csv"
    output_file_path = output_dir + output_file_name
    df.to_csv(output_file_path, index=False)
    print(f"Saved {output_file_name}")

def main():
    # File path to the Excel file
    file_path = f'{paths.data_path}/raw_data/demography/legal_partnership_by_lad.xlsx'

    # Process sheet 2 (female)
    process_sheet(file_path, sheet_name='2', sex='f')

    # Process sheet 3 (male)
    process_sheet(file_path, sheet_name='3', sex='m')

if __name__ == "__main__":
    main()