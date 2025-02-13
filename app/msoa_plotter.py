import folium
from folium.plugins import MarkerCluster
from tqdm import tqdm
from shapely.geometry import mapping
from object_identifiers import CareLocations
import streamlit as st

class MSOAPlotter:
    def __init__(self, data_loader, layer_options=None, center=None, zoom=None, selected_region=None, selected_LAD=None, selected_msoa=None):
        """
        If selected_msoa (and selected_region) are provided, the map will only show the selected MSOA
        boundary and its output area (OA) boundaries.
        """
        self.data_loader = data_loader
        self.layer_options = layer_options or {
            "hospitals": False,
            "hospices": False,
            "trusts": False,
            "care_homes": False,
        }
        self.center = center if center is not None else [54.5, -2.5]
        self.zoom = zoom if zoom is not None else 6
        self.map = folium.Map(location=self.center, zoom_start=self.zoom)
        self.cloc = CareLocations(self)
        self.selected_region = selected_region
        self.selected_LAD = selected_LAD
        self.selected_msoa = selected_msoa
        self.create_map()

    def create_map(self):
        m = self.map
        # If a specific MSOA is selected, only add its boundary and the OA boundaries within it.
        self.add_single_msoa_oa_boundaries_layer(m)
        folium.LayerControl(collapsed=False).add_to(m)

    def add_single_msoa_oa_boundaries_layer(self, m):
        """
        Add a layer showing only the output area (OA) boundaries that lie within the selected MSOA.
        This method uses the OA–MSOA–LAD mapping to extract the OA codes belonging to the selected MSOA.
        """
        # Assume that for each region the mapping DataFrame is indexed by the OA code.
        mapping_df = self.data_loader.df_filtered_hierarchy[self.selected_region]
        # Filter the mapping for rows where the 'msoa' column equals the selected MSOA.
        filtered = mapping_df[mapping_df['msoa'] == self.selected_msoa]
        if filtered.empty:
            st.write(f"No mapping found for MSOA: {self.selected_msoa} in region {self.selected_region}")
            return
        # Get the list of OA codes (assume the OA code is in a column named 'area' or the index).
        # Here, we assume the mapping has a column named 'area'.
        oa_codes = filtered['area'].unique().tolist()
        # Get the OA boundaries.
        gdf_oa = self.data_loader.oa_boundaries.get(self.selected_region)

        selected_msoa = gdf_oa[gdf_oa['OA21CD'].isin(oa_codes)]
        single_msoa_layer = folium.FeatureGroup(name=f"Output Areas for MSOA {self.selected_msoa}")
        for _, row in tqdm(selected_msoa.iterrows(),
                           total=selected_msoa.shape[0],
                           desc="Adding OA Boundaries for selected MSOA",
                           leave=False):
            properties = {
                "Region": self.selected_region,
                "LAD": self.selected_LAD,
                "MSOA": self.selected_msoa,
                "OA": row["OA21CD"],
            }
            feature = {
                "type": "Feature",
                "geometry": mapping(row["geometry"]),
                "properties": properties
            }
            gj = folium.GeoJson(
                data=feature,
                style_function=lambda x: {
                    "fillColor": "orange",
                    "color": "green",
                    "weight": 2,
                    "fillOpacity": 0.05,
                },
                highlight_function=lambda x: {"weight": 3, "fillOpacity": 0.5},
            )
            popup_content = self.generate_popup_content(properties)
            gj.add_child(folium.Popup(popup_content, max_width=450))
            gj.add_child(
                folium.GeoJsonTooltip(
                    fields=["Region", "LAD", "MSOA", "OA"],
                    aliases=["Region:", "LAD:", "MSOA:", "OA:"],
                    labels=False
                )
            )
            gj.add_to(single_msoa_layer)
        single_msoa_layer.add_to(m)
        # Adjust the map bounds to include these OA boundaries.
        bounds = selected_msoa.total_bounds  # [minx, miny, maxx, maxy]
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    @staticmethod
    def generate_popup_content(properties):
        content = ""
        for key, value in properties.items():
            content += f"<div><strong>{key}:</strong> {value}</div>"
        return content

def get_msoa_map(data_loader, layer_options, center, zoom, selected_region=None, selected_LAD=None, selected_msoa=None):
    mp = MSOAPlotter(data_loader, layer_options=layer_options, center=center, zoom=zoom,
                      selected_region=selected_region, selected_LAD=selected_LAD, selected_msoa=selected_msoa)
    return mp.map
