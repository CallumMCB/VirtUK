import streamlit as st
import pandas as pd
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
            with expander("Population Pyramids", expanded=True):
                # Retrieve demographic data for the MSOA.
                total_df = self.get_msoa_data(self.data.df_total_age, msoa)
                female_df = self.get_msoa_data(self.data.df_female_age, msoa)
                if total_df.empty or female_df.empty:
                    st.write("No demographic data available for this MSOA.")
                    return

                # Create sub-tabs for the two versions.
                sub_tab1, sub_tab2 = st.tabs(["Age by Single Year", "Age by 5 Year"])
                with sub_tab1:
                    chart_single = MSOA_Pop_summary.create_population_pyramid_single(self, total_df, female_df)
                    st.altair_chart(chart_single, use_container_width=True)
                with sub_tab2:
                    chart_five = MSOA_Pop_summary.create_population_pyramid_five(self, total_df, female_df)
                    st.altair_chart(chart_five, use_container_width=True)

    def create_population_pyramid_single(self, total_df, female_df):
        total_row = total_df.iloc[0]
        female_row = female_df.iloc[0]
        # Create a list of ages from 0 to 90.
        ages = list(range(0, 91))
        male_counts = []
        female_counts = []
        for age in ages:
            age_str = str(age)
            total_val = total_row.get(age_str, 0)
            female_val = female_row.get(age_str, 0)
            male_val = total_val - female_val
            male_counts.append(male_val)
            female_counts.append(female_val)
        # For the pyramid, make male counts negative.
        male_counts = [-x for x in male_counts]

        # Build a DataFrame. Create a string version for the y-axis.
        df_pyramid = pd.DataFrame({
            'age': ages,
            'age_str': [str(age) for age in ages],
            'male': male_counts,
            'female': female_counts
        })

        # Melt to long format.
        df_long = df_pyramid.melt(id_vars=['age', 'age_str'],
                                  value_vars=['male', 'female'],
                                  var_name='sex',
                                  value_name='count')

        # Create the bar chart. To have age 0 at the bottom and age 90 at the top,
        # we set the scale with reverse=True.
        chart = alt.Chart(df_long).mark_bar().encode(
            x=alt.X('count:Q',
                    title='Population',
                    axis=alt.Axis(labelExpr="abs(datum.value)")),
            y=alt.Y('age_str:N',
                    title='Age',
                    sort=[str(x) for x in ages],  # order as strings "0", "1", ..., "90"
                    scale=alt.Scale(domain=[str(x) for x in ages], reverse=True),
                    axis=alt.Axis(values=[str(x) for x in range(0, 91, 5)], grid=True)
                    ),
            color=alt.Color('sex:N',
                            title='Sex',
                            scale=alt.Scale(domain=['male', 'female'],
                                            range=['blue', 'red'])),
            tooltip=['age:Q', 'sex:N', 'count:Q']
        ).properties(
            width=400,
            height=400
        )

        # Add text labels above each bar (show absolute value).
        # text = alt.Chart(df_long).transform_calculate(
        #     abs_count="abs(datum.count)"
        # # ).mark_text(
        # #     align='center',
        # #     baseline='middle',
        # #     dy=-3,
        # #     fontSize=12,
        # #     color='black'
        # # ).encode(
        # #     x=alt.X('count:Q', axis=alt.Axis(labelExpr="abs(datum.value)")),
        # #     y=alt.Y('age_str:N', sort=[str(x) for x in ages],
        # #             scale=alt.Scale(domain=[str(x) for x in ages], reverse=True)),
        # #     text='abs_count:Q'
        # # )

        return chart

    def create_population_pyramid_five(self, total_df, female_df):
        total_row = total_df.iloc[0]
        female_row = female_df.iloc[0]
        # Define 5-year bins: 0-4, 5-9, ... 85-89, and 90+.
        bins = []
        bin_labels = []
        for start in range(0, 90, 5):
            bins.append((start, start + 4))
            bin_labels.append(f"{start}-{start + 4}")
        bins.append((90, 150))  # 90+ bracket; assume 150 is sufficiently high.
        bin_labels.append("90+")

        data_rows = []
        for i, (low, high) in enumerate(bins):
            label = bin_labels[i]
            ages_in_bin = [str(age) for age in range(low, high + 1)]
            total_sum = sum([total_row.get(age, 0) for age in ages_in_bin])
            female_sum = sum([female_row.get(age, 0) for age in ages_in_bin])
            male_sum = total_sum - female_sum
            # Compute the common amount and excess.
            common = min(male_sum, female_sum)
            male_excess = male_sum - common
            female_excess = female_sum - common

            # For males, values are negative.
            # The common part goes from 0 to -common.
            data_rows.append({
                'age_bin': label,
                'sex': 'male',
                'component': 'common',
                'value': -common,  # endpoint of common
                'abs_value': common,  # for tooltip if needed
                'total': male_sum
            })
            # For the excess part, start at -common and extend further left.
            data_rows.append({
                'age_bin': label,
                'sex': 'male',
                'component': 'excess',
                'baseline': -common,  # start at the end of the common segment
                'value': -common - male_excess,  # extend further left
                'excess': male_excess,  # absolute excess amount
                'total': male_sum
            })

            # For females, values are positive.
            # The common part goes from 0 to common.
            data_rows.append({
                'age_bin': label,
                'sex': 'female',
                'component': 'common',
                'value': common,
                'abs_value': common,
                'total': female_sum
            })
            # For the excess part, start at common and extend further right.
            data_rows.append({
                'age_bin': label,
                'sex': 'female',
                'component': 'excess',
                'baseline': common,  # start at the end of the common segment
                'value': common + female_excess,  # extend further right
                'excess': female_excess,  # absolute excess amount
                'total': female_sum
            })
        df_stack = pd.DataFrame(data_rows)

        # Create a column for color mapping.
        def fill_color(row):
            if row['sex'] == 'male' and row['component'] == 'common':
                return 'male_common'
            elif row['sex'] == 'male' and row['component'] == 'excess':
                return 'male_excess'
            elif row['sex'] == 'female' and row['component'] == 'common':
                return 'female_common'
            elif row['sex'] == 'female' and row['component'] == 'excess':
                return 'female_excess'

        df_stack['fill'] = df_stack.apply(fill_color, axis=1)

        # Build the chart for the common component.
        chart_common = alt.Chart(df_stack[df_stack.component == 'common']).mark_bar().encode(
            x=alt.X('value:Q',
                    title='Population',
                    axis=alt.Axis(labelExpr="abs(datum.value)")),
            y=alt.Y('age_bin:N',
                    title='Age',
                    sort=bin_labels,
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
        ).properties(
            width=400,
            height=400
        )

        # Build the chart for the excess component.
        # Use x and x2 to draw the bar starting at the baseline.
        chart_excess = alt.Chart(df_stack[df_stack.component == 'excess']).mark_bar().encode(
            x=alt.X('baseline:Q',
                    title='Population',
                    axis=alt.Axis(labelExpr="abs(datum.value)")),
            x2='value:Q',
            y=alt.Y('age_bin:N',
                    title='Age',
                    sort=bin_labels,
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
        ).properties(
            width=400,
            height=400
        )

        # Layer the two charts together.
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
