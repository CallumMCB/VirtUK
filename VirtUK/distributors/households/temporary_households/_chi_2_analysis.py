import pandas as pd
import numpy as np
from tabulate import tabulate
from scipy.stats import chisquare


class Chi2Analysis:
    """
    A class for running chi-squared analyses between produced (simulated) distributions
    and expected (target) distributions within a household simulation context.
    """

    def __init__(self, hh_instance, expected_values: pd.DataFrame):
        """
        Parameters
        ----------
        hh_instance : object
            An object containing household information (e.g., a manager or container
            of all simulated household data).
        expected_values : pd.DataFrame
            Target distribution values against which we compare the produced (simulated) values.
        """
        self.hh_instance = hh_instance
        self.expected_values = expected_values

        # Placeholder for storing the produced distribution DataFrame
        self.produced_distribution = None

    @classmethod
    def from_statistics(cls, hh_instance, statistics: pd.DataFrame, var_hh_composition: dict):
        """
        Create a Chi2Analysis instance using statistical data and variable household composition.

        Parameters
        ----------
        hh_instance : object
            Reference to the household instance / manager.
        statistics : pd.DataFrame
            Statistical distribution for the dimension(s) being analyzed (e.g., number of kids).
        var_hh_composition : dict
            A nested dictionary describing various household compositions (including 'd_family').

        Returns
        -------
        Chi2Analysis
            An initialized Chi2Analysis object with auto-computed expected_values.
        """
        # 1) Build a DataFrame from var_hh_composition
        df = pd.DataFrame.from_dict(var_hh_composition, orient="index")

        # 2) Sum the 'number' for rows where 'type' is 'd_family'
        d_family_sum = df.loc[df["type"] == "d_family", "number"].sum()

        # 3) Multiply the statistics by the total number of d_family households
        expected_values = d_family_sum * statistics

        return cls(hh_instance=hh_instance, expected_values=expected_values)

    def analyze_kids_distribution(self):
        """
        Gather information on how many kids live in specific household compositions
        (1-parent vs 2-parent), then compare with the expected distribution.

        This method:
            1. Identifies relevant households
            2. Categorizes the number of kids as 1, 2, or 3+ (capped at 3)
            3. Builds a produced distribution DataFrame
            4. Performs a chi-squared test comparing produced vs. expected distributions
        """
        # Step 1: Separate kids data for 1-parent vs 2-parent households
        one_parent_kids = []
        two_parent_kids = []

        # Simple helper function to cap number of kids at 3
        def categorize_kids(num_kids: int) -> int:
            return min(num_kids, 3)

        # Predefine recognized household compositions (in string form)
        composition_map = {
            "[1.5 0.5 0.5 1.  0. ]": one_parent_kids,
            "[1.5 0.5 0.5 2.  0. ]": two_parent_kids,
        }

        # Step 2: Loop over each household and assign the number of kids
        for hh in self.hh_instance.households:
            hh_comp_str = str(hh.composition)  # Convert composition to string
            if hh_comp_str in composition_map:
                # Get the number of kids in this household
                num_kids = hh.num_people["kid"]
                composition_map[hh_comp_str].append(categorize_kids(num_kids))

        # Step 3: Build produced distribution as a DataFrame
        # Convert raw lists to value counts, ensuring consistent indexing
        one_parent_counts = pd.Series(one_parent_kids).value_counts().sort_index()
        two_parent_counts = pd.Series(two_parent_kids).value_counts().sort_index()

        self.produced_distribution = pd.DataFrame({
            "1 parent": [
                one_parent_counts.get(1, 0),
                one_parent_counts.get(2, 0),
                one_parent_counts.get(3, 0),
            ],
            "2 parents": [
                two_parent_counts.get(1, 0),
                two_parent_counts.get(2, 0),
                two_parent_counts.get(3, 0),
            ],
        }, index=["1", "2", "3"])
        self.produced_distribution.index.name = "Number of Kids"

        # Step 4: Run comparison/chi-squared test
        self.compare_distributions(
            produced_df=self.produced_distribution,
            expected_df=self.expected_values,
            label="Kids Distribution"
        )

    def compare_distributions(
        self,
        produced_df: pd.DataFrame,
        expected_df: pd.DataFrame,
        label: str = "Distribution"
    ):
        """
        Compare the produced distribution against the expected distribution using chi-squared.

        Parameters
        ----------
        produced_df : pd.DataFrame
            DataFrame of produced (simulated) counts.
        expected_df : pd.DataFrame
            DataFrame of expected (target) counts, with the same row and column structure as produced_df.
        label : str, optional
            A label or name for the distribution being compared (printed in logs).

        Notes
        -----
        1. Adjusts the expected distribution if total sums differ (to allow for small mismatches).
        2. Runs a combined chi-squared test by flattening row-wise produced and expected arrays.
        """
        print(f"\n--- {label}: Produced vs. Expected ---")

        # Ensure the expected distribution is a DataFrame
        if not isinstance(expected_df, pd.DataFrame):
            expected_df = pd.DataFrame(expected_df)

        # If 'produced_df' has same columns as 'expected_df', rename them to align
        # Or just ensure columns match. If they don't, you'll need logic here to reorder columns.
        expected_df.index.name = produced_df.index.name

        # Print both side by side for clarity
        print("\nProduced Distribution:")
        print(tabulate(produced_df, headers="keys", tablefmt="pipe", showindex=True))
        print("\nExpected Distribution:")
        print(tabulate(expected_df, headers="keys", tablefmt="pipe", showindex=True))

        # Flatten the data for chi-squared
        combined_produced_values = []
        combined_expected_values = []

        for col in produced_df.columns:
            prod_vals = produced_df[col].values
            exp_vals = expected_df[col].values

            # Adjust expected values if row sums differ from produced
            total_produced = np.sum(prod_vals)
            total_expected = np.sum(exp_vals)
            if not np.isclose(total_produced, total_expected, rtol=1e-8):
                factor = total_produced / total_expected if total_expected != 0 else 1
                exp_vals = exp_vals * factor

            combined_produced_values.extend(prod_vals)
            combined_expected_values.extend(exp_vals)

        # Perform chi-squared test
        chi2, p = chisquare(combined_produced_values, f_exp=combined_expected_values)

        # Output the results
        print(f"\n[Chi-squared Test on '{label}']")
        print(f"  Chi-squared value: {chi2:.4f}")
        print(f"  p-value         : {p:.4g}")
        print("--------------------------------------------------")

    # ---------------------------------------------------------------------
    # Future Additional Methods:
    # def analyze_other_distribution(self, ...):
    #     """
    #     Example method for analyzing a different data distribution (e.g., age distribution,
    #     household size, etc.). You can pattern this after analyze_kids_distribution().
    #     """
    #     pass
    #
    # def compare_all(self):
    #     """
    #     Possibly run a loop or sequence of distribution analyses in one call.
    #     """
    #     pass
    # ---------------------------------------------------------------------
