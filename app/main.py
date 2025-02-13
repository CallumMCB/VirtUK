import os
import streamlit as st
import pandas as pd
import re
from streamlit_folium import st_folium
from caching_helpers import load_all_data, create_map, create_lad_map, create_msoa_map
from pprint import pprint
from outputs import MSOAOutput, OAOutput, CareOutput

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../data_2021"))

class Main:
    def __init__(self):
        self.start_center = [54.5, -2.5]
        self.start_zoom = 6
        self.regions_file = os.path.join(DATA_PATH, 'input/geography/oa_msoa_lad_regions.csv')
        self.lad_lookup = os.path.join(DATA_PATH, 'input/geography/lad_lookup.csv')
        self.selected_oa_code = None
        self.selected_msoa_code = None
        self.selected_LAD_code = None
        self.selected_region_name = None
        self.main()

    def main(self):
        st.set_page_config(page_title="VirtUK Interactive Map", layout="wide")
        st.title("VirtUK Interactive Map (2021 Census Day)")

        # --- Sidebar: Regions & Layer Options ---
        st.sidebar.header("Filters & Options")

        # Ensure `unique_regions` and `selected_regions` are always defined
        unique_regions = []  # Default empty list
        selected_regions = []  # Default empty list

        df_regions = pd.read_csv(self.regions_file)
        unique_regions = df_regions['region'].dropna().unique().tolist()
        unique_regions = [region for region in unique_regions if region != 'Wales']
        selected_regions = st.sidebar.multiselect("Select Regions", options=unique_regions, default=['North East'])
        regions = selected_regions if selected_regions else None

        df_lad_lookup = pd.read_csv(self.lad_lookup)
        df_lad_lookup.set_index('lad_code', inplace=True)

        # --- Load and Process Data (cached) ---
        data_loader = load_all_data(unique_regions, regions)

        st.sidebar.subheader("Feature Layers")
        feature_options = {
            "hospitals": st.sidebar.checkbox("Show Hospitals", value=False),
            "hospices": st.sidebar.checkbox("Show Hospices", value=False),
            "trusts": st.sidebar.checkbox("Show NHS Trusts", value=False),
            "care_homes": st.sidebar.checkbox("Show Care Homes", value=False),
        }
        st.sidebar.subheader("Boundary Layers")

        if "sidebar_config" not in st.session_state:
            st.session_state.sidebar_config = {
                "selected_regions": selected_regions,
                "feature_options": feature_options,
            }

        if 'map_center' not in st.session_state:
            st.session_state.map_center = [54.5, -2.5]
        if 'map_zoom' not in st.session_state:
            st.session_state.map_zoom = 6
        if 'last_location_clicked' not in st.session_state:
            st.session_state.last_location_clicked = None
        if 'last_area_clicked' not in st.session_state:
            st.session_state.last_area_clicked = None

        # --- Load and Process Data (cached) ---
        self.msoa_op = MSOAOutput(self, data_loader)
        self.oa_op = OAOutput(self, data_loader)  # Make sure OAOutput is defined!
        self.care_op = CareOutput(self)

        # --- Create Main Map ---
        self.mainmp, self.maininfo = st.columns([3, 2])
        with self.mainmp:
            st.subheader("Map of Selected Regions")
            main_map_data = st_folium(
                create_map(data_loader, feature_options, self.start_center, self.start_zoom),
                width=700,
                height=450
            )
            pprint(main_map_data)

        self.selected_LAD_code = self.extract_map_data(main_map_data, rf'[EW]0[6789]')

        # --- Layout: Two Columns (Map & Details) ---
        self.submpcol1, self.submpcol2 = st.columns([3, 2])
        if self.selected_LAD_code:
            with self.submpcol1:
                st.subheader(f"Map of {df_lad_lookup.loc[self.selected_LAD_code, 'lad']}")
                lad_map_data = st_folium(
                    create_lad_map(
                        data_loader,
                        feature_options,
                        self.start_center,
                        self.start_zoom,
                        self.selected_region_name,
                        self.selected_LAD_code
                    ),
                    width=700,
                    height=450
                )
                pprint(lad_map_data)

                self.selected_msoa_code = self.extract_map_data(lad_map_data, r'[EW]02')
                print(self.selected_msoa_code)
        # --- Display Sub-Map (MSOA Individual Map) in Column 2 ---
        if self.selected_msoa_code:
            with self.submpcol2:
                st.subheader("MSOA Individual Map")
                msoa_map_data = st_folium(
                    create_msoa_map(
                    data_loader,
                    feature_options,
                    self.start_center,
                    self.start_zoom,
                    self.selected_region_name,
                    self.selected_LAD_code,
                    self.selected_msoa_code
                ), width=700, height=450)
                pprint(msoa_map_data)

                self.selected_oa_code = self.extract_map_data(msoa_map_data, r'[EW]00')
                print(self.selected_oa_code)

        # --- Display Output Information in Columns ---
        msoa_oa_selector = st.popover("Select Area Information")
        show_oa = msoa_oa_selector.checkbox("Show Output Area", True)
        show_msoa = msoa_oa_selector.checkbox("Show MSOA", True)
        if show_oa and show_msoa:
            # Two outputs: create two columns side by side.
            col_left, col_right = st.columns(2)
            with col_left:
                if self.selected_msoa_code:
                    self.msoa_op.main_output(self.selected_msoa_code)
                else:
                    st.write("MSOA Not Selected or not available")
            with col_right:
                if self.selected_oa_code:
                    self.oa_op.main_output(self.selected_oa_code, self.selected_region_name)
                else:
                    st.write("OA Not Selected or not available")
        elif show_msoa:
            # Only MSOA output is selected.
            if self.selected_msoa_code:
                self.msoa_op.main_output(self.selected_msoa_code)
            else:
                st.write("MSOA Not Selected or not available")
        elif show_oa:
            # Only OA output is selected.
            if self.selected_oa_code:
                self.oa_op.main_output(self.selected_oa_code, self.selected_region_name)
            else:
                st.write("OA Not Selected or not available")
        else:
            st.write("No area selected")

    def extract_map_data(self, map_data, area_code_segment):
        # Extract MSOA code and region name from the tooltip text.
        if map_data.get("last_object_clicked_tooltip"):
            text = map_data["last_object_clicked_tooltip"]

            lines = [line.strip() for line in text.split("\n") if line.strip()]
            self.selected_region_name = lines[0] if lines else None

            area_match = re.search(rf"{area_code_segment}\d+", text)
            print(area_match)
            if area_match:
                return area_match.group()
            else:
                return None


if __name__ == "__main__":
    Main()
