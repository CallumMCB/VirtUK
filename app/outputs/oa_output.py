import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from streamlit import expander
from streamlit_folium import st_folium
import altair as alt
from pprint import pprint


class OAOutput:
    def __init__(self, main_container, data):
        # main_container should be an object with at least an attribute `col1`
        self.main = main_container
        self.oa_data = data

    def get_msoa_data(self, dataset, oa):
        # Here we filter on the 'area' column (the OA code)
        if isinstance(dataset, dict):
            for region, df in dataset.items():
                if oa in set(df['area']):  # Using a set for O(1) lookup
                    return df.loc[df['area'] == oa]
            return pd.DataFrame()  # Return an empty DataFrame if not found
        else:
            return dataset.loc[dataset['area'] == oa]

    def main_output(self, oa):
        st.subheader(f"OA: {oa}")
        # Create three tabs
        tab1, tab2, tab3 = st.tabs(["Population Summary", "Communal Residents", "Other"])
        OA_Pop_summary.main(self, tab1, oa)
        OA_Communal.main(self, tab2, oa)
        self.display_other(tab3, oa)

    def display_other(self, tab, oa):
        with tab:
            st.write("Other Data (Placeholder)")
            # Additional content for the 'Other' tab


class OA_Pop_summary:
    @staticmethod
    def main(self, tab, oa):
        with tab:
            st.subheader("Population Summary")
            # Retrieve the five‐year age data and the broad (student accommodation) age data
            five_year_df = self.get_msoa_data(self.oa_data.df_five_year_ages, oa)
            broad_age_df = self.get_msoa_data(self.oa_data.df_student_accommodation, oa)

            if five_year_df.empty or broad_age_df.empty:
                st.write(f"No population summary data available for OA: {oa}")
                return

            # --- Normalize Broad Age Data ---
            broad_age_columns = ['0-4', '5-15', '16-17', '18-20', '21-24', '25-29', '30+']
            for col in broad_age_columns:
                total = broad_age_df[col].sum()
                if total != 0:
                    broad_age_df[col] = broad_age_df[col] / total
                else:
                    broad_age_df[col] = 0

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
            five_year_bands = list(five_year_df.columns)
            if 'area' in five_year_bands:
                five_year_bands.remove('area')

            if 'accommodation type' not in broad_age_df.columns:
                st.write("Broad age data missing 'accommodation type' column.")
                return

            # Initialize a DataFrame to hold proportions per accommodation type.
            five_year_props = pd.DataFrame(
                0,
                index=broad_age_df['accommodation type'].unique(),
                columns=five_year_bands,
                dtype=float
            )

            # Distribute each broad age column’s proportion evenly among its mapped 5‑year bands.
            for broad_band, fband_list in broad_to_5y.items():
                if broad_band in broad_age_df.columns:
                    group_vals = broad_age_df.groupby('accommodation type')[broad_band].sum()
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

            # Normalize each 5‑year band so that the proportions sum to 1.
            five_year_props = five_year_props.div(five_year_props.sum(axis=0), axis=1)

            st.write("Five-year proportions:")

            # --- Estimate Counts ---
            # Get the total population counts for each 5‑year band (assumes one row per OA)
            pop_data = five_year_df.iloc[0][five_year_bands]
            estimated_counts = five_year_props.multiply(pop_data, axis=1)
            # "Non-student" is from the "Does not apply" row.
            non_student_counts = estimated_counts.loc['Does not apply']
            # "Students" are the remainder (sum over all other accommodation types)
            student_counts = estimated_counts.drop('Does not apply').sum()

            # --- Build a DataFrame for manual stacking (so we can use log scale) ---
            pop_df = pd.DataFrame({
                'Age Band': five_year_bands,
                'Students': [student_counts[band] for band in five_year_bands],
                'Non-Student': [non_student_counts[band] for band in five_year_bands]
            })
            pop_df['Total'] = pop_df['Students'] + pop_df['Non-Student']
            pop_df['NonStudentBottom'] = pop_df['Students']
            pop_df['NonStudentTop'] = pop_df['Total']
            pop_df['StudentsBottom'] = 0
            pop_df['StudentsTop'] = pop_df['Students']

            # Set up a y-axis domain (using log scale) based on the maximum total count.
            max_total = pop_df['Total'].max()
            y_domain = [1, max_total * 1.1]

            # --- Build the Altair charts ---
            # Chart for Students (the lower, blue portion)
            chart_students = alt.Chart(pop_df).mark_bar(color='#1f77b4').encode(
                x=alt.X('Age Band:N', sort=five_year_bands, title='5-Year Age Bands'),
                y=alt.Y('StudentsTop:Q', scale=alt.Scale(type='log', domain=y_domain), title='Estimated Count'),
                y2=alt.Y2('StudentsBottom:Q'),
                tooltip=[alt.Tooltip('Age Band:N'),
                         alt.Tooltip('Students:Q', title='Students Count')]
            )
            # Chart for Non-Students (stacked on top, in orange)
            chart_non_student = alt.Chart(pop_df).mark_bar(color='#ff7f0e').encode(
                x=alt.X('Age Band:N', sort=five_year_bands),
                y=alt.Y('NonStudentTop:Q', scale=alt.Scale(type='log', domain=y_domain)),
                y2=alt.Y2('NonStudentBottom:Q'),
                tooltip=[alt.Tooltip('Age Band:N'),
                         alt.Tooltip('Non-Student:Q', title='Non-Student Count')]
            )

            layered_chart = alt.layer(chart_students, chart_non_student).properties(
                title=f'Population by 5-Year Age Bands for {oa}',
                width=600,
                height=400
            ).interactive()

            st.altair_chart(layered_chart, use_container_width=True)

##########################################
# CLASS: OA_Communal (Interactive Altair)
##########################################

class OA_Communal:
    @staticmethod
    def main(self, tab, oa):
        with tab:
            st.subheader("Communal Residents")
            # Retrieve student accommodation data filtered by OA.
            student_df = self.get_msoa_data(self.oa_data.df_student_accommodation, oa)
            if student_df.empty:
                st.write(f"No communal resident data found for OA: {oa}")
                return

            # Define which accommodation types are considered communal.
            communal_types = ["communal establishment: Other", "communal establishment: University"]
            does_not_apply_type = "Does not apply"
            excluded_types = communal_types + [does_not_apply_type]
            age_cols = ["0-4", "5-15", "16-17", "18-20", "21-24", "25-29", "30+"]

            # Compute totals (summing across the age columns) for each group.
            communal_totals = student_df[student_df['accommodation type'].isin(communal_types)][age_cols].sum()
            does_not_apply_totals = student_df[student_df['accommodation type'] == does_not_apply_type][age_cols].sum()
            non_communal_totals = student_df[~student_df['accommodation type'].isin(excluded_types)][age_cols].sum()

            ########################################
            # Chart 1: Grouped Bar Chart
            ########################################
            grouped_data = []
            for cat in age_cols:
                grouped_data.append({
                    'Age Category': cat,
                    'Accom Category': 'Communal Student',
                    'Count': communal_totals[cat]
                })
                grouped_data.append({
                    'Age Category': cat,
                    'Accom Category': 'Non-Communal Student',
                    'Count': non_communal_totals[cat]
                })
                grouped_data.append({
                    'Age Category': cat,
                    'Accom Category': 'Not Student',
                    'Count': does_not_apply_totals[cat]
                })
            grouped_df = pd.DataFrame(grouped_data)

            grouped_chart = alt.Chart(grouped_df).mark_bar().encode(
                x=alt.X('Age Category:N', sort=age_cols, title='Age Categories'),
                # xOffset groups the bars side-by-side for each Age Category.
                xOffset=alt.X('Accom Category:N', sort=['Communal Student', 'Non-Communal Student', 'Not Student']),
                y=alt.Y('Count:Q',
                        scale=alt.Scale(type='log', domain=[1, grouped_df['Count'].max() * 1.1]),
                        title='Number of People'),
                color=alt.Color('Accom Category:N',
                                scale=alt.Scale(
                                    domain=['Communal Student', 'Non-Communal Student', 'Not Student'],
                                    range=['#FF5733', '#33C1FF', '#FFC300']),
                                title='Accom. Category'),
                tooltip=['Age Category', 'Accom Category', 'Count']
            ).properties(
                width=300,
                height=300,
                title='Residents by Accom. Type and Age (Grouped)'
            ).interactive()

            st.altair_chart(grouped_chart, use_container_width=True)

            ########################################
            # Chart 2: Stacked Bar Chart for Non-Communal Accommodation
            ########################################
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

            # Build the data for the stacked chart.
            stacked_rows = []
            for age_band in age_cols:
                start, end = age_ranges[age_band]
                cumulative = 0
                for ac in accom_types:
                    count = filtered_non_communal.loc[
                        filtered_non_communal['accommodation type'] == ac, age_band
                    ].sum()
                    stacked_rows.append({
                        'Age Band': age_band,
                        'Accom Type': ac,
                        'Count': count,
                        'Bottom': cumulative,
                        'Top': cumulative + count,
                        'AgeStart': start,
                        'AgeEnd': end
                    })
                    cumulative += count
            stacked_df = pd.DataFrame(stacked_rows)

            stacked_chart = alt.Chart(stacked_df).mark_bar().encode(
                x=alt.X('AgeStart:Q', title='Age', scale=alt.Scale(domain=[0, 35])),
                x2='AgeEnd:Q',
                y=alt.Y('Top:Q', title='Number of People'),
                y2='Bottom:Q',
                color=alt.Color('Accom Type:N',
                                scale=alt.Scale(
                                    domain=accom_types,
                                    range=[color_dict[ac] for ac in accom_types]),
                                title='Accommodation Type'),
                tooltip=['Age Band', 'Accom Type', 'Count']
            ).properties(
                width=300,
                height=300,
                title='Residents by Accom. Type and Age (Stacked)'
            ).interactive()

            st.altair_chart(stacked_chart, use_container_width=True)