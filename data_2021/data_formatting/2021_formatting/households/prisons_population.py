import pandas as pd
from fuzzywuzzy import process, fuzz
from VirtUK import paths

# -----------------------------------------------------------------------------
# File Paths and Sheet Settings
# -----------------------------------------------------------------------------
file_path = f"{paths.data_path}/raw_data/households/prison_population_data.xlsx"
output_dir = f'{paths.data_path}/input/households/communal_establishments/prisons'
sheet_name = "PT"

# (The code for reading and pivoting the Excel data is assumed to be the same as before.)
# For example:
data = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=7, usecols="B:D", header=None)
data.columns = ["Name", "Description", "Value"]
data["Name"].ffill(inplace=True)
data = data[data["Name"] != "Establishment"]
data["Description"] = data["Description"].astype(str)
pivot_df = data.pivot(index="Name", columns="Description", values="Value")

# -----------------------------------------------------------------------------
# Load Additional Prison Information
# -----------------------------------------------------------------------------
# This file contains extra details about each prison (including columns: Prison, area, msoa, etc.)
prisons_info = pd.read_csv(f"{output_dir}/prisons_formatted.csv")


# -----------------------------------------------------------------------------
# Merge Helper Using Fuzzy Matching with Longest Partial Match
# -----------------------------------------------------------------------------
def merge_with_prison_info(df, index_col="Name"):
    """
    Resets the index (so that prison names become a column) and, for each name,
    checks if an exact match exists in prisons_info["Prison"]. If not, it uses
    fuzzy matching. If multiple candidates tie for the best fuzzy score, the one
    with the highest fuzz.partial_ratio is selected. Only non exact matches
    (or those not matched at all) are recorded in a log table that is printed out.

    Finally, the function merges the df with prisons_info (adding columns 'area'
    and 'msoa') and returns the merged DataFrame.
    """
    df_reset = df.reset_index()
    prisons_list = prisons_info["Prison"].tolist()
    mapping = {}
    # List to store matching details for non exact (or no) matches.
    match_log = []

    for name in df_reset[index_col]:
        if name in prisons_list:
            # Exact match found; no logging required.
            mapping[name] = name
        else:
            # Use fuzzy matching to get up to 5 candidates.
            matches = process.extract(name, prisons_list, limit=5)
            if not matches:
                mapping[name] = None
                match_log.append({
                    "Original Name": name,
                    "Matched Candidate": None,
                    "Fuzzy Score": None,
                    "Partial Ratio": None,
                    "Note": "No match found"
                })
            else:
                best_score = matches[0][1]
                # Get all candidates with the best score.
                best_candidates = [m for m in matches if m[1] == best_score]
                if len(best_candidates) > 1:
                    # If tied, select the candidate with the highest partial ratio.
                    best_candidate = None
                    best_partial = -1
                    for candidate, score in best_candidates:
                        partial = fuzz.partial_ratio(name, candidate)
                        if partial > best_partial:
                            best_partial = partial
                            best_candidate = candidate
                    mapping[name] = best_candidate
                    match_log.append({
                        "Original Name": name,
                        "Matched Candidate": best_candidate,
                        "Fuzzy Score": best_score,
                        "Partial Ratio": best_partial,
                        "Note": "Multiple candidates; longest partial match selected"
                    })
                else:
                    # Single best candidate found.
                    # Optionally enforce a minimum threshold (e.g., 80) for acceptance.
                    if best_score < 80:
                        mapping[name] = None
                        match_log.append({
                            "Original Name": name,
                            "Matched Candidate": None,
                            "Fuzzy Score": best_score,
                            "Partial Ratio": fuzz.partial_ratio(name, matches[0][0]),
                            "Note": "Low score match; not used"
                        })
                    else:
                        mapping[name] = matches[0][0]
                        match_log.append({
                            "Original Name": name,
                            "Matched Candidate": matches[0][0],
                            "Fuzzy Score": best_score,
                            "Partial Ratio": fuzz.partial_ratio(name, matches[0][0]),
                            "Note": "Fuzzy match used"
                        })

    # Create a new column "Matched_Prison" using the mapping.
    df_reset["Matched_Prison"] = df_reset[index_col].map(mapping)
    # Merge with prisons_info to add 'area' and 'msoa'.
    merged = df_reset.merge(prisons_info[['Prison', 'area', 'msoa']],
                            left_on="Matched_Prison", right_on="Prison", how="left")
    merged.drop(columns=["Prison", "Matched_Prison"], inplace=True)

    # Convert the match_log list to a DataFrame.
    if match_log:
        match_log_df = pd.DataFrame(match_log)
        # Print only non-exact matches (those recorded in the log).
        print("Final non-exact or unmatched results:")
        print(match_log_df)
    else:
        print("All prison names matched exactly.")

    return merged

def remove_code_prefix(col):
    parts = col.split(" ", 1)
    if len(parts) == 2:
        return parts[1]
    return col

# -----------------------------------------------------------------------------
# Process Age Groups Table
# -----------------------------------------------------------------------------
age_groups_list = [
    "15 - 17", "18 - 20", "21 - 24", "25 - 29", "30 - 39",
    "40 - 49", "50 - 59", "60 - 69", "70 and over"
]
age_group_cols = [col for col in pivot_df.columns if col in age_groups_list]
age_group_df = pivot_df[age_group_cols].fillna("*")
merged_age_group = merge_with_prison_info(age_group_df)
merged_age_group.to_csv(f"{output_dir}/age_group_table.csv", index=False)

# -----------------------------------------------------------------------------
# Process Custody Types Table
# -----------------------------------------------------------------------------
custody_types_list = ["AA Remand", "D Sentenced", "L Non-criminal"]
custody_cols = [col for col in pivot_df.columns if col in custody_types_list]
# Remove the code prefix from the column names.
custody_df = pivot_df[custody_cols].rename(columns=remove_code_prefix).fillna("*")
merged_custody = merge_with_prison_info(custody_df)
merged_custody.to_csv(f"{output_dir}/custody_type_table.csv", index=False)

# -----------------------------------------------------------------------------
# Process Nationality Groups Table
# -----------------------------------------------------------------------------
nationality_groups_list = ["A British Nationals", "B Foreign Nationals", "C Not Recorded"]
nationality_cols = [col for col in pivot_df.columns if col in nationality_groups_list]
# Remove the code prefix from the column names.
nationality_df = pivot_df[nationality_cols].rename(columns=remove_code_prefix).fillna("*")
merged_nationality = merge_with_prison_info(nationality_df)
merged_nationality.to_csv(f"{output_dir}/nationality_group_table.csv", index=False)

# -----------------------------------------------------------------------------
# Process Offence Groups Table
# -----------------------------------------------------------------------------
offence_groups_list = [
    "01 Violence against the person", "02 Sexual offences", "03 Burglary", "04 Theft offences",
    "05 Criminal damage and arson", "06 Drug offences", "07 Possession of weapons",
    "08 Public order offences", "09 Miscellaneous crimes against society", "10 Fraud offences",
    "11 Summary offences", "12 Offence not recorded"
]
offence_cols = [col for col in pivot_df.columns if col in offence_groups_list]
# Remove the code prefix from the column names.
offence_df = pivot_df[offence_cols].rename(columns=remove_code_prefix).fillna("*")
merged_offence = merge_with_prison_info(offence_df)
merged_offence.to_csv(f"{output_dir}/offence_group_table.csv", index=False)

# -----------------------------------------------------------------------------
# Process Ethnicity Groups Table
# -----------------------------------------------------------------------------
ethnicity_groups_list = [
    "A Asian / Asian British", "B Black / African / Caribbean / Black British",
    "C Mixed / Multiple ethnic groups", "D Other ethnic group", "E White",
    "F Not recorded", "G Not stated"
]
ethnicity_cols = [col for col in pivot_df.columns if col in ethnicity_groups_list]
# For ethnicity, we keep the original column names.
ethnicity_df = pivot_df[ethnicity_cols].fillna("*")
merged_ethnicity = merge_with_prison_info(ethnicity_df)
merged_ethnicity.to_csv(f"{output_dir}/ethnicity_group_table.csv", index=False)

