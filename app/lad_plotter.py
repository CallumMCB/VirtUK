import folium
from folium.plugins import MarkerCluster
from tqdm import tqdm
from shapely.geometry import mapping
from object_identifiers import CareLocations
import streamlit as st

class LADPlotter:
    def __init__(self, data_loader, layer_options=None, center=None, zoom=None, selected_region=None, selected_LAD=None, selected_LAD_name=None):
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
        self.selected_LAD_name = selected_LAD_name
        self.create_map()

    def create_map(self):
        m = self.map
        self.add_single_lad_msoa_boundaries_layer(m)
        folium.LayerControl(collapsed=False).add_to(m)

    def add_single_lad_msoa_boundaries_layer(self, m):
        """
        Add a layer showing only the MSOA boundaries that lie within the selected LAD.
        This method uses the OA–MSOA–LAD mapping to extract the MSOA codes belonging to the selected LAD.
        """
        # Assume that for each region the mapping DataFrame is indexed by the OA code.
        mapping_df = self.data_loader.df_filtered_hierarchy[self.selected_region]
        # Filter the mapping for rows where the 'msoa' column equals the selected MSOA.
        filtered = mapping_df[mapping_df['lad_code'] == self.selected_LAD]
        # Get the list of MSOA codes.
        # Here, we assume the mapping has a column named 'msoa'.
        msoa_codes = filtered['msoa'].unique().tolist()
        # Get the OA boundaries.
        gdf_msoa = self.data_loader.msoa_boundaries.get(self.selected_region)

        selected_LAD = gdf_msoa[gdf_msoa['MSOA21CD'].isin(msoa_codes)]
        single_LAD_layer = folium.FeatureGroup(name=f"Output Areas for MSOA {self.selected_LAD}")
        for _, row in tqdm(selected_LAD.iterrows(),
                           total=selected_LAD.shape[0],
                           desc="Adding MSOA Boundaries for selected LAD",
                           leave=False):
            properties = {
                "Region": self.selected_region,
                "LAD": self.selected_LAD_name,
                "MSOA": row["MSOA21CD"],
            }
            feature = {
                "type": "Feature",
                "geometry": mapping(row["geometry"]),
                "properties": properties
            }
            gj = folium.GeoJson(
                data=feature,
                style_function=lambda x: {
                    "fillColor": "yellow",
                    "color": "red",
                    "weight": 2,
                    "fillOpacity": 0.05,
                },
                highlight_function=lambda x: {"weight": 3, "fillOpacity": 0.5},
            )
            popup_content = self.generate_popup_content(properties)
            gj.add_child(folium.Popup(popup_content, max_width=450))
            gj.add_child(
                folium.GeoJsonTooltip(
                    fields=["Region", "LAD", "MSOA"],
                    aliases=["Region:", "LAD Code:", "MSOA:"],
                    labels=False
                )
            )
            gj.add_to(single_LAD_layer)
        single_LAD_layer.add_to(m)
        # Adjust the map bounds to include these MSOA boundaries.
        bounds = selected_LAD.total_bounds  # [minx, miny, maxx, maxy]
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    @staticmethod
    def generate_popup_content(properties):
        content = ""
        for key, value in properties.items():
            content += f"<div><strong>{key}:</strong> {value}</div>"
        return content

def get_LAD_map(data_loader, layer_options, center, zoom, selected_region=None, selected_LAD=None, selected_LAD_name=None):
    mp = LADPlotter(data_loader, layer_options=layer_options, center=center, zoom=zoom,
                      selected_region=selected_region, selected_LAD=selected_LAD, selected_LAD_name=selected_LAD_name)
    return mp.map
