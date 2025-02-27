import streamlit as st

from data_loader import DataLoader, DataFilter
from map_plotter import get_map
from lad_plotter import get_LAD_map
from msoa_plotter import get_msoa_map

@st.cache_data(show_spinner=True)
def load_all_data(unique_regions, regions):
    if regions == None:
        regions = unique_regions
    data_loader = DataLoader(regions)
    DataFilter(data_loader, regions=regions)
    return data_loader

# We do not cache map creation because folium objects are not picklable.
def create_map(data_loader, layer_options, center, zoom):
    return get_map(data_loader, layer_options, center, zoom)

def create_lad_map(data_loader, layer_options, center, zoom, region, lad, lad_name):
    return get_LAD_map(data_loader, layer_options, center, zoom, region, lad, lad_name)

def create_msoa_map(data_loader, layer_options, center, zoom, region, lad, msoa, msoa_name):
    return get_msoa_map(data_loader, layer_options, center, zoom, region, lad, msoa, msoa_name)