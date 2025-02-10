import streamlit as st
import pandas as pd
import numpy as np
from streamlit import expander
import altair as alt

class MSOAOutput:
    def __init__(self, main_container, data_loader):
        self.main = main_container
        self.data = data_loader

    def get_msoa_data(self, dataset, msoa):
        if isinstance(dataset, dict):
            for region, df in dataset.items():
                if msoa in set(df['msoa']):
                    return df.loc[df['msoa'] == msoa]
            return pd.DataFrame()
        else:
            return dataset.loc[dataset['msoa'] == msoa]

    def main_output(self, msoa):
        st.subheader(f"MSOA: {msoa}")
        # Create three tabs
        tab1, tab2, tab3 = st.tabs(["Population Summary", "Communal Residents", "Other"])
        MSOA_Pop_summary.main(self, tab1, msoa)
        MSOA_Communal.main(self, tab2, msoa)
        self.display_other(tab3, msoa)

    def display_other(self, tab, msoa):
        with tab:
            st.write("Other Data (Placeholder)")
            # Additional content for the 'Other'

class MSOA_Pop_summary(MSOAOutput):
    def main(self, tab, msoa):
        with tab:
            st.subheader('Population Summary')
            with st.expander("Population Pyramids", expanded=True):
                # Retrieve demographic data for the MSOA.
                total_df = self.get_msoa_data(self.data.df_total_age, msoa)
                female_df = self.get_msoa_data(self.data.df_female_age, msoa)
                if total_df.empty or female_df.empty:
                    st.write("No demographic data available for this MSOA.")
                    return

                # Precompute population data once.
                df_population = MSOA_Pop_summary.precompute_population_data(self, total_df, female_df)

                # Create sub-tabs for the two versions and the table.
                sub_tab1, sub_tab2, sub_tab3 = st.tabs([
                    "Age by Single Year", "Age by 5 Year", "Data Table"
                ])

                # Chart for single year pyramid.
                with sub_tab1:
                    chart_single = MSOA_Pop_summary.create_population_pyramid_single(self, df_population)
                    st.altair_chart(chart_single, use_container_width=True)

                # Chart for five year pyramid.
                with sub_tab2:
                    chart_five = MSOA_Pop_summary.create_population_pyramid_five(self, df_population)
                    st.altair_chart(chart_five, use_container_width=True)

                # Display a table of single year population data.
                with sub_tab3:
                    st.dataframe(df_population.set_index('age'))

    def precompute_population_data(self, total_df, female_df):
        """
        Precompute a single–year population table with columns:
          - age
          - male (computed as total - female)
          - female
          - total

        Assumes that there is only one row in the msoa data.
        Vectorises over ages 0–90.
        """
        # Get the only row corresponding to the MSOA.
        total_row = total_df.iloc[0]
        female_row = female_df.iloc[0]

        # Create an array of ages and matching string labels (to match DataFrame columns).
        ages = np.arange(0, 91)
        age_labels = [str(age) for age in ages]

        # Reindex each Series over these age labels and fill missing values with 0.
        total_vals = total_row.reindex(age_labels, fill_value=0).astype(int)
        female_vals = female_row.reindex(age_labels, fill_value=0).astype(int)

        # Precompute male counts.
        male_vals = total_vals - female_vals

        # Build and return the DataFrame.
        df_single = pd.DataFrame({
            'age': ages,
            'male': male_vals.values,     # positive counts for the table
            'female': female_vals.values,
            'total': total_vals.values
        })
        return df_single

    def create_population_pyramid_single(self, df_population):
        """
        Create an Altair bar chart for the population pyramid using single–year data.
        (For the chart, male counts are made negative.)
        """
        # Copy the precomputed data for charting and negate the male counts.
        df_chart = df_population.copy()
        df_chart['male'] = -df_chart['male']
        df_chart['age_str'] = df_chart['age'].astype(str)  # for proper categorical ordering on the y-axis

        # Convert to long format.
        df_long = df_chart.melt(
            id_vars=['age', 'age_str'],
            value_vars=['male', 'female'],
            var_name='sex',
            value_name='count'
        )

        # Build the bar chart.
        chart = alt.Chart(df_long).mark_bar().encode(
            x=alt.X('count:Q',
                    title='Population',
                    axis=alt.Axis(labelExpr="abs(datum.value)")),
            y=alt.Y('age_str:N',
                    title='Age',
                    sort=[str(x) for x in range(0, 91)],
                    scale=alt.Scale(domain=[str(x) for x in range(0, 91)], reverse=True),
                    axis=alt.Axis(values=[str(x) for x in range(0, 91, 5)], grid=True)
                    ),
            color=alt.Color('sex:N',
                            title='Sex',
                            scale=alt.Scale(
                                domain=['male', 'female'],
                                range=['blue', 'red']
                            )),
            tooltip=['age:Q', 'sex:N', 'count:Q']
        ).properties(
            width=400,
            height=400
        )
        return chart

    def create_population_pyramid_five(self, df_population):
        """
        Create an Altair layered bar chart for the five–year population pyramid.
        This method groups the precomputed single–year data into 5–year bins and then
        computes, for each bin:
          - The "common" count (the minimum of male and female counts)
          - The "excess" counts (the remainder for each sex)
        """
        # Work on a copy of the precomputed data.
        df_single = df_population.copy()

        # Define bins: 0-4, 5-9, …, 85-89, and 90+.
        bins = list(range(0, 91, 5)) + [np.inf]
        labels = [f"{i}-{i+4}" for i in range(0, 90, 5)] + ["90+"]

        # Assign each age to a bin.
        df_single['age_bin'] = pd.cut(df_single['age'], bins=bins, right=False, labels=labels)

        # Group by age_bin and sum up male, female, and total counts.
        df_bins = df_single.groupby('age_bin', as_index=False, observed=False).sum()

        # Compute the common count (shared by both sexes) and the excess for each.
        df_bins['common'] = df_bins[['male', 'female']].min(axis=1)
        df_bins['male_excess'] = df_bins['male'] - df_bins['common']
        df_bins['female_excess'] = df_bins['female'] - df_bins['common']

        # Build the stacked data for the pyramid.
        # For males, common counts will be negative.
        male_common = df_bins[['age_bin', 'common']].copy()
        male_common['sex'] = 'male'
        male_common['component'] = 'common'
        male_common['baseline'] = 0
        male_common['value'] = -male_common['common']
        male_common = male_common.rename(columns={'common': 'abs_value'})
        male_common['total'] = df_bins['male']

        male_excess = df_bins[['age_bin', 'male_excess', 'common']].copy()
        male_excess['sex'] = 'male'
        male_excess['component'] = 'excess'
        male_excess['baseline'] = -male_excess['common']
        male_excess['value'] = -male_excess['common'] - male_excess['male_excess']
        male_excess = male_excess.rename(columns={'male_excess': 'excess'})
        male_excess['total'] = df_bins['male']

        # For females, the common counts remain positive.
        female_common = df_bins[['age_bin', 'common']].copy()
        female_common['sex'] = 'female'
        female_common['component'] = 'common'
        female_common['baseline'] = 0
        female_common['value'] = female_common['common']
        female_common = female_common.rename(columns={'common': 'abs_value'})
        female_common['total'] = df_bins['female']

        female_excess = df_bins[['age_bin', 'female_excess', 'common']].copy()
        female_excess['sex'] = 'female'
        female_excess['component'] = 'excess'
        female_excess['baseline'] = female_excess['common']
        female_excess['value'] = female_excess['common'] + female_excess['female_excess']
        female_excess = female_excess.rename(columns={'female_excess': 'excess'})
        female_excess['total'] = df_bins['female']

        # Combine all parts.
        df_stack = pd.concat(
            [male_common, male_excess, female_common, female_excess],
            ignore_index=True
        )

        # Create a helper column for fill colors.
        conditions = [
            (df_stack['sex'] == 'male') & (df_stack['component'] == 'common'),
            (df_stack['sex'] == 'male') & (df_stack['component'] == 'excess'),
            (df_stack['sex'] == 'female') & (df_stack['component'] == 'common'),
            (df_stack['sex'] == 'female') & (df_stack['component'] == 'excess'),
        ]
        choices = ['male_common', 'male_excess', 'female_common', 'female_excess']
        df_stack['fill'] = np.select(conditions, choices, default='unknown')

        # Build the chart for the common component.
        chart_common = alt.Chart(
            df_stack[df_stack['component'] == 'common']
        ).mark_bar().encode(
            x=alt.X('value:Q',
                    title='Population',
                    axis=alt.Axis(labelExpr="abs(datum.value)")),
            y=alt.Y('age_bin:N',
                    title='Age',
                    sort=labels,
                    scale=alt.Scale(reverse=True),
                    axis=alt.Axis(grid=True)),
            color=alt.Color('fill:N',
                            scale=alt.Scale(
                                domain=['male_common', 'female_common'],
                                range=['lightblue', 'lightcoral']
                            ),
                            legend=alt.Legend(title="Component")),
            tooltip=[
                alt.Tooltip('age_bin:N', title='Age Bin'),
                alt.Tooltip('sex:N', title='Sex'),
                alt.Tooltip('total:Q', title='Total')
            ]
        ).properties(width=400, height=400)

        # Build the chart for the excess component using x2.
        chart_excess = alt.Chart(
            df_stack[df_stack['component'] == 'excess']
        ).mark_bar().encode(
            x=alt.X('baseline:Q',
                    title='Population',
                    axis=alt.Axis(labelExpr="abs(datum.value)")),
            x2='value:Q',
            y=alt.Y('age_bin:N',
                    title='Age',
                    sort=labels,
                    scale=alt.Scale(reverse=True),
                    axis=alt.Axis(grid=True)),
            color=alt.Color('fill:N',
                            scale=alt.Scale(
                                domain=['male_excess', 'female_excess'],
                                range=['darkblue', 'darkred']
                            ),
                            legend=alt.Legend(title="Component")),
            tooltip=[
                alt.Tooltip('age_bin:N', title='Age Bin'),
                alt.Tooltip('sex:N', title='Sex'),
                alt.Tooltip('total:Q', title='Total'),
                alt.Tooltip('excess:Q', title='Excess Amount', format=".0f")
            ]
        ).properties(width=400, height=400)

        # Layer the two charts.
        layered_chart = alt.layer(chart_common, chart_excess).resolve_scale(
            color='independent'
        )
        return layered_chart

class MSOA_Communal(MSOAOutput):
    def main(self, tab, msoa):
        with tab:
            st.subheader("Communal Residents")
            with expander("Overall Summary", expanded=True):
                types = MSOA_Communal.communal_residents(self, tab, msoa)
                MSOA_Communal.communal_staff(self, tab, msoa)
                MSOA_Communal.plot_communal_residents_chart(self, tab, msoa)
            MSOA_Communal.display_cqc_data(self, tab, msoa)

    def communal_residents(self, tab, msoa):
        # Retrieve resident type data for this MSOA using our helper.
        type_df = self.get_msoa_data(self.data.df_resident_type_msoa, msoa)
        if type_df.empty:
            with tab:
                st.write(f"No communal resident data found for MSOA: {msoa}")
            return

        # Rename columns for better display.
        rename = {
            'MC: Care home with nursing': 'Care Home: Nursing',
            'MC: Care home without nursing': 'Care Home: Non-Nursing',
            "MC: Children's home": "Children's Home",
            "MC: General hospital": "General Hospital",
            'MC: Mental health hospital or unit': 'Mental Health Hospital',
            'MC: Other hospital': 'Other Hospital',
            'MC: Home or hostel': 'Home or Hospital',
            'MC: Other home': 'Other Home',
            'Other: Approved premises': 'Approved Premises',
            'Other: Detention centres': 'Detention Centres',
            'Other: Education': 'Education',
            'Other: Prison service': 'Prison Service',
            'Other: Religious': 'Religious',
            'Other: Staff/Worker accommodation': 'Staff/Worker Accommodation',
            'Other: Temporary accommodation': 'Temporary Accommodation',
        }
        # Remove any columns that are all zeros, then rename.
        nonzero_type_df = type_df.loc[:, (type_df != 0).any(axis=0)].rename(columns=rename)

        st.dataframe(nonzero_type_df)

    def communal_staff(self, tab, msoa):
        staff_df = self.get_msoa_data(self.data.df_staff_or_temporary, msoa)
        if staff_df.empty:
            st.write(f"No staff data found for MSOA: {msoa}")
        else:
            st.write("ONS 2021 Census Data")
            st.dataframe(staff_df)

    def plot_communal_residents_chart(self, tab, msoa):
        # Retrieve female and male resident data for this MSOA.
        female_df = self.get_msoa_data(self.data.df_female_residents, msoa)
        male_df = self.get_msoa_data(self.data.df_male_residents, msoa)
        if female_df.empty or male_df.empty:
            st.write(f"Insufficient resident data for MSOA: {msoa}")
            return

        # Assuming one row per MSOA (adjust if necessary).
        female_series = female_df.iloc[0]
        male_series = male_df.iloc[0]

        # Define the original age bin names (as in your data).
        old_age_bins = [
            "Aged 0 to 15",
            "Aged 16 to 24",
            "Aged 25 to 34",
            "Aged 35 to 49",
            "Aged 50 to 64",
            "Aged 65 to 74",
            "Aged 75 to 84",
            "Aged 85 and over"
        ]

        # Define the new age bin labels in the format "lower-higher"
        new_age_bins = [
            "0-15",
            "16-24",
            "25-34",
            "35-49",
            "50-64",
            "65-74",
            "75-84",
            "85+"
        ]

        # Extract the relevant columns using the old names.
        female_series = female_series[old_age_bins]
        male_series = male_series[old_age_bins]

        # Rename the series indices to the new bin names.
        female_series.index = new_age_bins
        male_series.index = new_age_bins

        # Combine the series into a DataFrame.
        df_combined = pd.DataFrame({
            'Female': female_series,
            'Male': male_series
        }, index=new_age_bins)
        df_combined = df_combined.reindex(new_age_bins)

        # Optionally filter to only include bins that are nonzero.
        mask = (df_combined['Female'] != 0) | (df_combined['Male'] != 0)
        nonzero_bins = df_combined.index[mask].tolist()
        if nonzero_bins:
            start_pos = new_age_bins.index(nonzero_bins[0])
            end_pos = new_age_bins.index(nonzero_bins[-1])
            df_combined = df_combined.iloc[start_pos:end_pos + 1]

        # Define the color scale.
        color_scale = alt.Scale(domain=["Female", "Male"], range=["red", "blue"])

        # Convert to long format for Altair.
        df_long = df_combined.reset_index().melt(
            id_vars='index',
            var_name='Sex',
            value_name='Count'
        ).rename(columns={'index': 'AgeBand'})

        # Create a grouped bar chart with horizontal labels and tick marks.
        chart = alt.Chart(df_long).mark_bar().encode(
            x=alt.X('AgeBand:N',
                    title='Age Band',
                    sort=list(df_combined.index),
                    axis=alt.Axis(labelAngle=0, tickSize=5)),
            y=alt.Y('Count:Q', title='Number of Residents'),
            color=alt.Color('Sex:N', title='Sex', scale=color_scale),
            tooltip=['AgeBand:N', 'Sex:N', 'Count:Q'],
            xOffset=alt.X('Sex:N')
        ).properties(
            width=600,
            height=400
        )

        # Create text labels above each bar.
        text = alt.Chart(df_long).mark_text(
            align='center',
            baseline='bottom',
            dy=-3,  # vertical offset (adjust as needed)
            fontSize=12,
            color='black'
        ).encode(
            x=alt.X('AgeBand:N',
                    title='Age Band',
                    sort=list(df_combined.index),
                    axis=alt.Axis(labelAngle=0, tickSize=5)),
            xOffset=alt.X('Sex:N'),
            y=alt.Y('Count:Q'),
            text='Count:Q'
        )

        # Layer the text on top of the bars.
        final_chart = chart + text

        st.write("Communal Residents")
        st.altair_chart(final_chart, use_container_width=True)

    def display_cqc_data(self, tab, msoa):
        # Mapping for nicer column names.
        CQC_columns = {
            "Location Name": "Name",
            "Location Postal Code": "Postcode",
            "area": "Output Area",
            "Service type - Care home service with nursing": "Nursing",
            "Service type - Care home service without nursing": "Non-Nursing",
            "Care homes beds": "Beds",
            "occupancy_mean": "Mean Occupancy",
            "occupancy_std": "Occupancy std.",
            "Service user band - Children 0-18 years": "Children",
            "Service user band - Younger Adults": "Youger Adults",
            "Service user band - Older People": "Older Adults"
        }
        cqc_df = self.get_msoa_data(self.data.df_care_homes, msoa)
        if cqc_df.empty:
            st.write(f"No CQC data found for MSOA: {msoa}")
            return
        cqc_df = cqc_df[list(CQC_columns.keys())]
        cqc_df = cqc_df.rename(columns=CQC_columns).set_index("Name")

        ons_df = self.get_msoa_data(self.data.df_resident_type_msoa, msoa)
        if ons_df.empty:
            st.write(f"No ONS data found for MSOA: {msoa}")
            return
        ons_df = ons_df[['msoa', 'MC: Care home with nursing', 'MC: Care home without nursing']]
        ons_df = ons_df.rename(columns={
            'MC: Care home with nursing': 'Nursing Occupants',
            'MC: Care home without nursing': 'Non-Nursing Occupants'
        }).set_index('msoa')
        try:
            ons_df = ons_df.loc[msoa]
        except KeyError:
            st.write(f"MSOA {msoa} not found in ONS data.")
            return

        if not (cqc_df.empty and ons_df.empty) and (ons_df.values.sum() != 0):
            with st.expander("Care Homes"):
                st.write("ONS Data")
                st.dataframe(ons_df)
                st.write("CQC Data")
                st.dataframe(cqc_df)
