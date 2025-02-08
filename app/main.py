import os
import streamlit as st
import pandas as pd
import re
from streamlit_folium import st_folium
from caching_helpers import load_all_data, create_map, create_sub_map
from pprint import pprint
from outputs import MSOAOutput, OAOutput, CareOutput  # Ensure OAOutput exists!

DATA_PATH = 'data_2021'

class Main:
    def __init__(self):
        self.start_center = [54.5, -2.5]
        self.start_zoom = 6
        self.regions_file = os.path.join(DATA_PATH, 'input/geography/oa_msoa_lad_regions.csv')
        self.selected_oa_code = None
        self.selected_msoa_code = None
        self.selected_region_name = None
        self.main()

    def main(self):
        st.set_page_config(page_title="PoppyPop & JUNE Map", layout="wide")
        st.title("PoppyPop & JUNE Map (2021 Census Day)")

        # --- Sidebar: Regions & Layer Options ---
        st.sidebar.header("Filters & Options")
        if os.path.exists(self.regions_file):
            df_regions = pd.read_csv(self.regions_file)
            unique_regions = df_regions['region'].unique().tolist()
            selected_regions = st.sidebar.multiselect("Select Regions", options=unique_regions, default=['North East'])
            regions = selected_regions if selected_regions else None
        else:
            regions = None

        st.sidebar.subheader("Feature Layers")
        feature_options = {
            "hospitals": st.sidebar.checkbox("Show Hospitals", value=False),
            "hospices": st.sidebar.checkbox("Show Hospices", value=False),
            "trusts": st.sidebar.checkbox("Show NHS Trusts", value=False),
            "care_homes": st.sidebar.checkbox("Show Care Homes", value=True),
        }
        st.sidebar.subheader("Boundary Layers")
        boundary_options = {
            "msoa_boundaries": st.sidebar.checkbox("Show MSOA Boundaries", value=True),
            "lad_boundaries": st.sidebar.checkbox("Show LAD Boundaries", value=False),
        }
        layer_options = {**feature_options, **boundary_options}

        if "sidebar_config" not in st.session_state:
            st.session_state.sidebar_config = {
                "selected_regions": selected_regions,
                "feature_options": feature_options,
                "boundary_options": boundary_options,
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
        data_loader = load_all_data(regions)
        self.msoa_op = MSOAOutput(self, data_loader)
        self.oa_op = OAOutput(self, data_loader)  # Make sure OAOutput is defined!
        self.care_op = CareOutput(self)

        # --- Layout: Two Columns (Map & Details) ---
        self.col1, self.col2 = st.columns([3, 2])
        with self.col1:
            st.subheader("Map")
            main_map_data = st_folium(
                create_map(data_loader, layer_options, self.start_center, self.start_zoom),
                width=700,
                height=450
            )
            pprint(main_map_data)

        # --- Process Clicks from the Main Map ---
        if main_map_data.get("last_object_clicked"):
            clicked = main_map_data["last_object_clicked"]

        if main_map_data.get("last_active_drawing"):
            clicked_area = main_map_data["last_active_drawing"]
            if "properties" in clicked_area:
                st.session_state.last_area_clicked = clicked_area["properties"]
                print("Clicked area from main map:", clicked_area)

        # Extract MSOA code and region name from the tooltip text.
        if main_map_data.get("last_object_clicked_tooltip"):
            text = main_map_data["last_object_clicked_tooltip"]
            # Extract the MSOA code using regex (e.g. "E02xxxxx")
            msoa_match = re.search(r"E02\d+", text)
            if msoa_match:
                self.selected_msoa_code = msoa_match.group()
            else:
                self.selected_msoa_code = None
            # Extract the region name as the first non-empty line.
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            self.selected_region_name = lines[0] if lines else None

        # --- Display Sub-Map (MSOA Individual Map) in Column 2 ---
        if self.selected_msoa_code:
            with self.col2:
                st.subheader("MSOA Individual Map")
                msoa_map = create_sub_map(
                    data_loader,
                    layer_options,
                    self.start_center,
                    self.start_zoom,
                    self.selected_region_name,
                    self.selected_msoa_code
                )
                sub_map_data = st_folium(msoa_map, width=700, height=450)
                pprint(sub_map_data)
                if sub_map_data.get("last_object_clicked"):
                    sub_clicked = sub_map_data["last_object_clicked"]
                    print("Clicked point on sub-map:", sub_clicked)
                if sub_map_data.get("last_active_drawing"):
                    sub_clicked_area = sub_map_data["last_active_drawing"]
                    if "properties" in sub_clicked_area:
                        st.session_state.last_area_clicked_sub = sub_clicked_area["properties"]
                        print("Clicked area on sub-map:", sub_clicked_area)
                if sub_map_data.get("last_object_clicked_tooltip"):
                    sub_tooltip = sub_map_data["last_object_clicked_tooltip"]
                    self.selected_oa_code = sub_tooltip.strip()
                    print("Sub-map tooltip:", self.selected_oa_code)

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
                    self.oa_op.main_output(self.selected_oa_code)
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
                self.oa_op.main_output(self.selected_oa_code)
            else:
                st.write("OA Not Selected or not available")
        else:
            st.write("No area selected")

if __name__ == "__main__":
    Main()
