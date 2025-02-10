import streamlit as st
import pandas as pd
from tabulate import tabulate
from streamlit import expander
import altair as alt
import re

alt.renderers.set_embed_options(vegaLiteVersion="5")

# -----------------------------------------------------------------------------
# Helper: Extract the starting number from an age band for sorting.
# If the band starts with "<", return 0.
# -----------------------------------------------------------------------------
def extract_age(band):
    if band.startswith('<'):
        return 0
    m = re.search(r'\d+', band)
    if m:
        return int(m.group())
    return float('inf')

# ============================================================================
# Base Class: OAOutput
# ============================================================================
class OAOutput:
    def __init__(self, main_container, data):
        self.oa = None
        self.region = None
        self.main = main_container
        self.data = data

    def main_output(self, oa, region):
        self.oa = oa
        self.region = region
        self.oa_data()
        st.subheader(f"OA: {self.oa}")
        # Create three tabs
        tab1, tab2, tab3 = st.tabs(["Population Summary", "Communal Residents", "Other"])
        OA_Pop_summary.main(self, tab1)
        OA_Communal.main(self, tab2)
        OA_Households.main(self, tab3)

    def oa_data(self):
        self.five_year_df = self.get_oa_data(self.data.df_five_year_ages)
        self.broad_age_df = self.get_oa_data(self.data.df_student_accommodation)
        self.student_df = self.get_oa_data(self.data.df_student_accommodation)
        self.prisons_df = self.get_oa_data(self.data.df_prisons_formatted)

    def get_oa_data(self, dataset):
        """
        Original non-cached method. In many cases you might use the cached
        version instead.
        """
        df = dataset[self.region]
        return df.loc[df['area'] == self.oa].copy()

    def display_other(self, tab):
        with tab:
            st.write("Other Data (Placeholder)")


# ============================================================================
# CLASS: OA_Pop_summary (Population Summary)
# ============================================================================
class OA_Pop_summary(OAOutput):
    def main(self, tab):
        with tab:
            st.subheader("Population Summary")
            # Retrieve the five‐year age data and the broad (student accommodation) age data
            OA_Pop_summary.student_v_normal_population(self)

    def student_v_normal_population(self):
        if  self.five_year_df.empty or self.broad_age_df.empty:
            st.write(f"No population summary data available for OA: {self.oa}")
            return

        # --- Normalize Broad Age Data ---
        broad_age_columns = ['0-4', '5-15', '16-17', '18-20', '21-24', '25-29', '30+']
        for col in broad_age_columns:
            # Convert column to float before dividing.
            self.broad_age_df[col] = self.broad_age_df[col].astype(float)
            total = self.broad_age_df[col].sum()
            if total != 0:
                self.broad_age_df.loc[:, col] = self.broad_age_df.loc[:, col] / total
            else:
                self.broad_age_df.loc[:, col] = 0.0

        # --- Map Broad Age Bands to 5-Year Bands ---
        broad_to_5y = {
            '0-4': ['<4'],
            '5-15': ['5-9', '10-14'],
            '16-17': ['15-19'],
            '18-20': ['15-19'],
            '21-24': ['20-24'],
            '25-29': ['25-29'],
            '30+': ['30-34', '35-39']
        }

        # Assume the five-year DataFrame has one column with the OA code (e.g. "area")
        # and then one column per 5‑year band.
        five_year_bands = list( self.five_year_df.columns)
        if 'area' in five_year_bands:
            five_year_bands.remove('area')
        # Sort the five-year bands in ascending order (based on the first number)
        sorted_five_year_bands = sorted(five_year_bands, key=extract_age)

        if 'accommodation type' not in self.broad_age_df.columns:
            st.write("Broad age data missing 'accommodation type' column.")
            return

        # Initialize a DataFrame to hold proportions per accommodation type.
        five_year_props = pd.DataFrame(
            0,
            index=self.broad_age_df['accommodation type'].unique(),
            columns=five_year_bands,
            dtype=float
        )

        # Distribute each broad age column’s proportion evenly among its mapped 5‑year bands.
        for broad_band, fband_list in broad_to_5y.items():
            if broad_band in self.broad_age_df.columns:
                group_vals = self.broad_age_df.groupby('accommodation type')[broad_band].sum()
                split_factor = len(fband_list)
                for fband in fband_list:
                    five_year_props.loc[group_vals.index, fband] += group_vals / split_factor
            else:
                st.write(f"Warning: Broad band '{broad_band}' not found in data.")

        # For any 5‑year bands that were not mapped, assign them to "Does not apply"
        remaining_bands = ['40-44', '45-49', '50-54', '55-59', '60-64',
                           '65-69', '70-74', '75-79', '80-84', '85>']
        if 'Does not apply' not in five_year_props.index:
            five_year_props.loc['Does not apply'] = 0
        for band in remaining_bands:
            five_year_props.loc['Does not apply', band] = 1.0

        # Normalize each 5‑year band so that the proportions sum to 1,
        # and fill NaNs that might occur if the column sum is zero.
        five_year_props = five_year_props.div(five_year_props.sum(axis=0), axis=1).fillna(0)

        # --- Estimate Counts ---
        pop_data =  self.five_year_df.iloc[0][five_year_bands]
        estimated_counts = five_year_props.multiply(pop_data, axis=1)

        # Compute student and non-student counts.
        non_student_counts = estimated_counts.loc['Does not apply']
        student_counts = estimated_counts.drop('Does not apply').sum()

        # Build DataFrame for plotting.
        pop_df = pd.DataFrame({
            'Age Band': sorted_five_year_bands,
            'Students': [student_counts[band] for band in sorted_five_year_bands],
            'Non-Student': [non_student_counts[band] for band in sorted_five_year_bands]
        })
        pop_df['Total'] = pop_df['Students'] + pop_df['Non-Student']

        # Fill any possible NaN values before rounding/casting.
        pop_df = pop_df.fillna(0)
        pop_df['Students'] = pop_df['Students'].round().astype(int)
        pop_df['Non-Student'] = pop_df['Non-Student'].round().astype(int)
        pop_df['Total'] = pop_df['Total'].round().astype(int)

        # --- Build the Altair stacked bar chart with a legend ---
        # Convert pop_df to long format.
        pop_long_df = pop_df.melt(
            id_vars=['Age Band', 'Total'],
            value_vars=['Students', 'Non-Student'],
            var_name='Type',
            value_name='Count'
        )

        y_domain = [0, pop_df['Total'].max() * 1.1]

        chart = alt.Chart(pop_long_df).mark_bar().encode(
            x=alt.X('Age Band:N', sort=sorted_five_year_bands, title='5-Year Age Bands'),
            y=alt.Y('Count:Q', scale=alt.Scale(domain=y_domain), title='Estimated Count'),
            color=alt.Color('Type:N',
                            scale=alt.Scale(domain=["Students", "Non-Student"],
                                            range=["#1f77b4", "#ff7f0e"]),
                            title="Category"),
            tooltip=['Age Band', 'Type', 'Count']
        ).properties(
            title=f'Population by 5-Year Age Bands for {self.oa}',
            width=600,
            height=400
        ).interactive()

        # --- Create two tabs: one for the chart and one for the data table ---
        chart_tab, data_tab = st.tabs(["Chart", "Data"])
        with chart_tab:
            st.altair_chart(chart, use_container_width=True)
        with data_tab:
            # Set the Age Band as index and then add a total row.
            pop_df = pop_df.set_index("Age Band")
            total_row = pop_df.sum(numeric_only=True)
            total_row.name = "Total"
            pop_df = pd.concat([pop_df, pd.DataFrame(total_row).T])
            st.dataframe(pop_df)

# ============================================================================
# CLASS: OA_Communal (Communal Residents)
# ============================================================================

class OA_Communal(OAOutput):
    def main(self, tab):
        with tab:
            st.subheader("Communal Residents")
            # Display student accommodation data.
            with st.expander("Student Data"):
                OA_Communal.students_communal(self)
            # Now show prison data.
            OA_Communal.prisons(self)

    def prisons(self):
        with st.expander("Prison Data"):
            # Assuming self.prisons_df is a DataFrame with a "Prison" column.
            self.prisons_df.set_index("Prison", inplace=True)
            st.write("**ONS 2021 Census Recording**")
            st.dataframe(self.prisons_df)
            st.caption("The 'total communal residents' and the 'prison service' values relate to the total number of communal residents in the output area and the total number of residents in the prison service in the msoa respectively."
                       "\nOne will note that this number is often significantly lower than recorded by the prison service itself (below). This is due to the fact that the Census only registers those residing here for >12 months.")

            # Get the processed prisons actual dictionary for the current region.
            prisons_actual = self.data.df_prisons_actual[self.region]

            # --- Age Group Charts ---
            st.write("**Age Group Data**")
            age_group_df = prisons_actual['age_group']
            # Filter to the current output area.
            age_group_df = age_group_df[age_group_df['area'] == self.oa]
            age_cols = ["15 - 17", "18 - 20", "21 - 24", "25 - 29", "30 - 39",
                        "40 - 49", "50 - 59", "60 - 69", "70 and over"]
            for idx, row in age_group_df.iterrows():
                chart = OA_Communal.create_prison_age_group_chart(self, row, age_cols)
                st.altair_chart(chart, use_container_width=True)

            # --- Custody Type Data (Display as Table) ---
            st.write("**Custody Type Data**")
            custody_df = prisons_actual['custody_type']
            custody_df = custody_df[custody_df['area'] == self.oa]
            st.dataframe(custody_df)

            # --- Nationality Group Data (Display as Table) ---
            st.write("**Nationality Group Data**")
            nationality_df = prisons_actual['nationality_group']
            nationality_df = nationality_df[nationality_df['area'] == self.oa]
            st.dataframe(nationality_df)

            # --- Offence Group Pie Chart with Tabs ---
            st.write("**Offence Group Data**")
            offence_df = prisons_actual['offence_group']
            offence_df = offence_df[offence_df['area'] == self.oa]
            offence_cols = [col for col in offence_df.columns if col not in ['area', 'msoa', 'Name']]
            for idx, row in offence_df.iterrows():
                # Here we request the underlying data to be returned.
                pie_chart, underlying_df = OA_Communal.create_prison_pie_chart(self, row, offence_cols,
                                                title=f"Offence Distribution for {row.get('Name', 'Unknown')}",
                                                return_data=True)
                chart_tab, data_tab = st.tabs(["Chart", "Data"])
                with chart_tab:
                    st.altair_chart(pie_chart, use_container_width=True)
                with data_tab:
                    st.dataframe(underlying_df)

            # --- Ethnicity Group Pie Chart with Tabs ---
            st.write("**Ethnicity Group Data**")
            ethnicity_df = prisons_actual['ethnicity_group']
            ethnicity_df = ethnicity_df[ethnicity_df['area'] == self.oa]
            ethnicity_cols = [col for col in ethnicity_df.columns if col not in ['area', 'msoa', 'Name']]
            for idx, row in ethnicity_df.iterrows():
                pie_chart, underlying_df = OA_Communal.create_prison_pie_chart(self, row, ethnicity_cols,
                                                title=f"Ethnicity Distribution for {row.get('Name', 'Unknown')}",
                                                return_data=True)
                chart_tab, data_tab = st.tabs(["Chart", "Data"])
                with chart_tab:
                    st.altair_chart(pie_chart, use_container_width=True)
                with data_tab:
                    st.dataframe(underlying_df)

    def students_communal(self):
        """
        Creates a grouped bar chart showing counts by age category for:
          - Communal Student (University)
          - Communal Student (Other)
          - Non-Communal Student
          - Not Student
        """
        communal_univ = "communal establishment: University"
        communal_other = "communal establishment: Other"
        does_not_apply = "Does not apply"
        excluded_types = [communal_univ, communal_other, does_not_apply]
        age_cols = ["0-4", "5-15", "16-17", "18-20", "21-24", "25-29", "30+"]

        communal_univ_totals = self.student_df[self.student_df['accommodation type'] == communal_univ][age_cols].sum()
        communal_other_totals = self.student_df[self.student_df['accommodation type'] == communal_other][age_cols].sum()
        does_not_apply_totals = self.student_df[self.student_df['accommodation type'] == does_not_apply][age_cols].sum()
        non_communal_totals = self.student_df[~self.student_df['accommodation type'].isin(excluded_types)][age_cols].sum()

        if communal_univ_totals.sum() + communal_other_totals.sum() == 0:
            st.write("There are no communal students in this output area.")
            return

        grouped_data = []
        for cat in age_cols:
            grouped_data.append({
                'Age Category': cat,
                'Type Category': 'Communal Student (University)',
                'Count': communal_univ_totals[cat]
            })
            grouped_data.append({
                'Age Category': cat,
                'Type Category': 'Communal Student (Other)',
                'Count': communal_other_totals[cat]
            })
            grouped_data.append({
                'Age Category': cat,
                'Type Category': 'Non-Communal Student',
                'Count': non_communal_totals[cat]
            })
            grouped_data.append({
                'Age Category': cat,
                'Type Category': 'Not Student',
                'Count': does_not_apply_totals[cat]
            })

        grouped_df = pd.DataFrame(grouped_data)
        grouped_df['Count'] = grouped_df['Count'].round().astype(int)
        grouped_chart = alt.Chart(grouped_df).mark_bar().encode(
            x=alt.X(
                'Age Category:N',
                sort=sorted(age_cols, key=extract_age),
                title='Age Categories',
                scale=alt.Scale(paddingOuter=0.2),
                axis=alt.Axis(labelAngle=0, tickBand='extent')
            ),
            xOffset=alt.X(
                'Type Category:N',
                sort=[
                    'Communal Student (University)',
                    'Communal Student (Other)',
                    'Non-Communal Student',
                    'Not Student'
                ],
                bandPosition=0.2,
                scale=alt.Scale(paddingInner=0.1, paddingOuter=0.3)
            ),
            y=alt.Y(
                'Count:Q',
                scale=alt.Scale(domain=[0, grouped_df['Count'].max() * 1.1]),
                title='Number of People'
            ),
            color=alt.Color(
                'Type Category:N',
                scale=alt.Scale(scheme='category10'),
                title='Type Category'
            ),
            tooltip=['Age Category', 'Type Category', 'Count']
        ).properties(
            width=300,
            height=300,
            title='Student vs Non-Student'
        ).interactive()

        chart_tab, data_tab = st.tabs(["Chart", "Data"])
        with chart_tab:
            st.altair_chart(grouped_chart, use_container_width=True)
        with data_tab:
            pivot_communal = grouped_df.pivot(index='Type Category', columns='Age Category', values='Count')
            pivot_communal = pivot_communal.fillna(0).astype(int)
            sorted_age_cols = sorted(pivot_communal.columns, key=extract_age)
            pivot_communal = pivot_communal.reindex(columns=sorted_age_cols)
            pivot_communal["Total"] = pivot_communal.sum(axis=1)
            total_row = pivot_communal.sum(numeric_only=True)
            total_row.name = "Total"
            pivot_communal = pd.concat([pivot_communal, pd.DataFrame(total_row).T])
            st.dataframe(pivot_communal)

    @staticmethod
    def create_prison_age_group_chart(self, prison_row, age_cols):
        data = []
        for col in age_cols:
            original_val = prison_row[col]
            try:
                numeric_val = 0 if original_val == "*" else float(original_val)
            except Exception:
                numeric_val = 0
            data.append({
                "Age Range": col,
                "Value": numeric_val,
                "Original": original_val
            })
        df_chart = pd.DataFrame(data)
        chart = alt.Chart(df_chart).mark_bar().encode(
            x=alt.X('Age Range:N', sort=age_cols, title='5-Year Age Band'),
            y=alt.Y('Value:Q', title='Estimated Count'),
            tooltip=[alt.Tooltip('Age Range:N', title='Age Range'),
                     alt.Tooltip('Original:N', title='Count')]
        ).properties(
            title=prison_row.get('Name', 'Prison Age Distribution')
        )
        return chart

    @staticmethod
    def create_prison_bar_chart(self, row, cols, title=""):
        data = []
        for col in cols:
            original_val = row[col]
            try:
                numeric_val = 0 if original_val == "*" else float(original_val)
            except Exception:
                numeric_val = 0
            data.append({
                "Category": col,
                "Value": numeric_val,
                "Original": original_val
            })
        df_chart = pd.DataFrame(data)
        chart = alt.Chart(df_chart).mark_bar().encode(
            x=alt.X('Category:N', sort=cols, title="Category"),
            y=alt.Y('Value:Q', title="Value"),
            tooltip=[alt.Tooltip('Category:N', title="Category"),
                     alt.Tooltip('Original:N', title="Value")]
        ).properties(
            title=title if title else row.get("Name", "Prison Chart")
        )
        return chart

    @staticmethod
    def create_prison_pie_chart(self, row, cols, title="", alternate=False, return_data=False):
        """
        Create a pie chart from a row of data.
        Any "*" values are treated as 0 (for the angle) but are shown in the tooltip.
        If return_data is True, the method returns a tuple of (chart, underlying_data_df).
        This method is used for both offence group and ethnicity group.
        """
        data = []
        for col in cols:
            original_val = row[col]
            try:
                numeric_val = 0 if original_val == "*" else float(original_val)
            except Exception:
                numeric_val = 0
            data.append({
                "Category": col,
                "Value": numeric_val,
                "Original": original_val
            })
        df = pd.DataFrame(data)
        chart = alt.Chart(df).mark_arc(innerRadius=30).encode(
            theta=alt.Theta(field="Value", type="quantitative"),
            color=alt.Color(field="Category", type="nominal"),
            tooltip=[alt.Tooltip('Category:N', title="Category"),
                     alt.Tooltip('Original:N', title="Value")]
        ).properties(
            title=title if title else row.get("Name", "Distribution")
        )
        if return_data:
            return chart, df
        else:
            return chart

# ============================================================================
# CLASS: OA_Households (Households)
# ============================================================================
class OA_Households(OAOutput):
    def main(self, tab):
        with tab:
            st.subheader("Households")
            with expander("Student's Housing (Non-Communal)"):
                # Retrieve student accommodation data using the cached function.
                OA_Households.students_non_communal(self, self.student_df)

    def students_non_communal(self, student_df):
        """
        Creates a stacked density chart for non-communal accommodation types,
        showing number density (count per unit age) across broad age bands.
        """
        # Define communal accommodation types for filtering.
        communal_univ = "communal establishment: University"
        communal_other = "communal establishment: Other"
        does_not_apply = "Does not apply"
        excluded_types = [communal_univ, communal_other, does_not_apply]
        # Define the age columns.
        age_cols = ["0-4", "5-15", "16-17", "18-20", "21-24", "25-29", "30+"]

        # Filter out communal and "Does not apply" rows.
        filtered_non_communal = student_df[~student_df['accommodation type'].isin(excluded_types)]

        # Define numerical age ranges corresponding to the broad age bands.
        age_ranges = {
            '0-4': (0, 5),
            '5-15': (5, 16),
            '16-17': (16, 18),
            '18-20': (18, 21),
            '21-24': (21, 25),
            '25-29': (25, 30),
            '30+': (30, 35)
        }

        # Get the unique non-communal accommodation types and assign colors.
        accom_types = sorted(filtered_non_communal['accommodation type'].unique())
        color_list = ["#008000", "#FF69B4", "#FFA500", "#66023C"]
        color_dict = {ac: color_list[i % len(color_list)] for i, ac in enumerate(accom_types)}

        # Build the data for the stacked density chart.
        stacked_rows = []
        for age_band in age_cols:
            start, end = age_ranges[age_band]
            bin_width = end - start
            cumulative_density = 0  # Used for stacking.
            for ac in accom_types:
                # Get the raw count for this accommodation type in the given age band.
                count = filtered_non_communal.loc[
                    filtered_non_communal['accommodation type'] == ac, age_band
                ].sum()
                # Compute the density: count per unit age.
                density = count / bin_width if bin_width else 0
                stacked_rows.append({
                    'Age Band': age_band,
                    'Accom Type': ac,
                    'Count': count,
                    'Density': density,
                    'BottomDensity': cumulative_density,
                    'TopDensity': cumulative_density + density,
                    'AgeStart': start,
                    'AgeEnd': end,
                    'BinWidth': bin_width
                })
                cumulative_density += density

        stacked_df = pd.DataFrame(stacked_rows)
        # Round the counts to the nearest integer.
        stacked_df['Count'] = stacked_df['Count'].round().astype(int)

        # Compute an appropriate y-axis domain for the density values.
        max_top_density = stacked_df['TopDensity'].max()

        # Determine unique bin edges from age_ranges.
        bin_edges = sorted({edge for r in age_ranges.values() for edge in r})

        # Create the Altair density chart.
        stacked_chart = alt.Chart(stacked_df).mark_bar().encode(
            x=alt.X('AgeStart:Q', title='Age',
                    scale=alt.Scale(domain=[0, 35]),
                    axis=alt.Axis(values=bin_edges, labelAngle=0)),
            x2='AgeEnd:Q',
            y=alt.Y('TopDensity:Q', title='Number Density (count per unit age)',
                    scale=alt.Scale(domain=[0, max_top_density * 1.1])),
            y2='BottomDensity:Q',
            color=alt.Color('Accom Type:N',
                            scale=alt.Scale(
                                domain=accom_types,
                                range=[color_dict[ac] for ac in accom_types]),
                            title='Accommodation Type'),
            tooltip=['Age Band', 'Accom Type', 'Count', 'Density']
        ).properties(
            width=300,
            height=300,
            title='Non-Communal Students by Accommodation Type'
        ).configure_title(
            offset=20
        ).interactive()

        # --- Create two tabs: one for the chart and one for the pivoted data table ---
        chart_tab, data_tab = st.tabs(["Chart", "Data"])
        with chart_tab:
            st.altair_chart(stacked_chart, use_container_width=True)
            st.caption("Note: The height of each bar is a density (count per unit age).")
        with data_tab:
            # Pivot the data so that rows are accommodation types and columns are age brackets.
            pivot_student = stacked_df.pivot(index='Accom Type', columns='Age Band', values='Count')
            pivot_student = pivot_student.fillna(0).astype(int)
            # Reorder the age columns in ascending order.
            sorted_age_cols = sorted(pivot_student.columns, key=extract_age)
            pivot_student = pivot_student.reindex(columns=sorted_age_cols)
            # Add a total column (summing across rows).
            pivot_student["Total"] = pivot_student.sum(axis=1)
            # Add a total row (summing each column).
            total_row = pivot_student.sum(numeric_only=True)
            total_row.name = "Total"
            pivot_student = pd.concat([pivot_student, pd.DataFrame(total_row).T])
            st.dataframe(pivot_student)
