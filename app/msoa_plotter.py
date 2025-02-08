import folium
from folium.plugins import MarkerCluster
from tqdm import tqdm
from shapely.geometry import mapping
from object_identifiers import CareLocations
import streamlit as st

class MSOAPlotter:
    def __init__(self, data_loader, layer_options=None, center=None, zoom=None, selected_region=None, selected_msoa=None):
        """
        If selected_msoa (and selected_region) are provided, the map will only show the selected MSOA
        boundary and its output area (OA) boundaries.
        """
        self.data_loader = data_loader
        self.layer_options = layer_options or {
            "hospitals": True,
            "hospices": True,
            "trusts": True,
            "care_homes": False,
            "msoa_boundaries": True,
            "lad_boundaries": False
        }
        self.center = center if center is not None else [54.5, -2.5]
        self.zoom = zoom if zoom is not None else 6
        self.map = folium.Map(location=self.center, zoom_start=self.zoom)
        self.cloc = CareLocations(self)
        self.selected_region = selected_region
        self.selected_msoa = selected_msoa
        self.create_map()

    def create_map(self):
        m = self.map
        # If a specific MSOA is selected, only add its boundary and the OA boundaries within it.
        if self.selected_msoa and self.selected_region:
            self.add_single_msoa_boundary_layer(m)
            self.add_single_msoa_oa_boundaries_layer(m)
        folium.LayerControl(collapsed=False).add_to(m)

    def add_single_msoa_boundary_layer(self, m):
        """
        Add a layer showing only the boundary of the selected MSOA.
        """
        poly = None
        # Look up the selected MSOA polygon in msoa_boundaries.
        if isinstance(self.data_loader.msoa_boundaries, dict):
            # Assume the dictionary key matching the selected region.
            gdf = self.data_loader.msoa_boundaries.get(self.selected_region)
            if gdf is not None:
                candidate = gdf[gdf["MSOA21CD"] == self.selected_msoa]
                if not candidate.empty:
                    poly = candidate.iloc[0]["geometry"]
        else:
            candidate = self.data_loader.msoa_boundaries[
                self.data_loader.msoa_boundaries["MSOA21CD"] == self.selected_msoa
            ]
            if not candidate.empty:
                poly = candidate.iloc[0]["geometry"]
        if poly is None:
            st.write(f"MSOA boundary not found for {self.selected_msoa}")
            return
        msoa_boundary_layer = folium.FeatureGroup(name=f"MSOA {self.selected_msoa} Boundary")
        feature = {
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {"MSOA": self.selected_msoa}
        }
        gj = folium.GeoJson(
            data=feature,
            style_function=lambda x: {
                "fillColor": "blue",
                "color": "red",
                "weight": 3,
                "fillOpacity": 0
            },
            highlight_function=lambda x: {"weight": 4, "fillOpacity": 0.3}
        )
        gj.add_child(folium.Popup(f"MSOA: {self.selected_msoa}", max_width=450))
        msoa_boundary_layer.add_child(gj)
        msoa_boundary_layer.add_to(m)
        # Adjust the map bounds to the MSOA boundary.
        bounds = poly.bounds  # (minx, miny, maxx, maxy)
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    def add_single_msoa_oa_boundaries_layer(self, m):
        """
        Add a layer showing only the output area (OA) boundaries that lie within the selected MSOA.
        This method uses the OA–MSOA–LAD mapping (assumed stored as a dictionary keyed by region in
        data_loader.df_filtered_hierarchy) to extract the OA codes belonging to the selected MSOA.
        """
        # Ensure the mapping hierarchy for the region is available.
        if not hasattr(self.data_loader, "df_filtered_hierarchy"):
            st.write("Mapping hierarchy not loaded.")
            return
        if self.selected_region not in self.data_loader.df_filtered_hierarchy:
            st.write(f"Mapping hierarchy not available for region: {self.selected_region}")
            return
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
        if isinstance(self.data_loader.oa_boundaries, dict):
            gdf_oa = self.data_loader.oa_boundaries.get(self.selected_region)
            if gdf_oa is None:
                st.write(f"No OA boundaries available for region: {self.selected_region}")
                return
        else:
            gdf_oa = self.data_loader.oa_boundaries
        selected_oa = gdf_oa[gdf_oa['OA21CD'].isin(oa_codes)]
        if selected_oa.empty:
            st.write(f"No OA boundaries found for MSOA: {self.selected_msoa}")
            return
        single_oa_layer = folium.FeatureGroup(name=f"Output Areas for MSOA {self.selected_msoa}")
        for _, row in tqdm(selected_oa.iterrows(),
                           total=selected_oa.shape[0],
                           desc="Adding OA Boundaries for selected MSOA",
                           leave=False):
            properties = {
                "layer_type": "oa boundary for msoa",
                "OA": row["OA21CD"],
                "MSOA": self.selected_msoa,
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
                    fields=["OA"],
                    aliases=["OA:"],
                    labels=False
                )
            )
            gj.add_to(single_oa_layer)
        single_oa_layer.add_to(m)
        # Adjust the map bounds to include these OA boundaries.
        bounds = selected_oa.total_bounds  # [minx, miny, maxx, maxy]
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    def add_msoa_boundaries_layer(self, m):
        # (Global MSOA boundaries as in your original code)
        if isinstance(self.data_loader.msoa_boundaries, dict):
            for region, gdf in self.data_loader.msoa_boundaries.items():
                region_layer = folium.FeatureGroup(name=f"MSOA Boundaries - {region}")
                for _, row in tqdm(gdf.iterrows(),
                                   total=gdf.shape[0],
                                   desc=f"Adding MSOA Boundaries for {region}",
                                   leave=False):
                    properties = {
                        "layer_type": "msoa boundary",
                        "MSOA": row["MSOA21CD"],
                        "region": region
                    }
                    feature = {
                        "type": "Feature",
                        "geometry": mapping(row["geometry"]),
                        "properties": properties
                    }
                    gj = folium.GeoJson(
                        data=feature,
                        style_function=lambda x: {
                            "fillColor": "blue",
                            "color": "red",
                            "weight": 2,
                            "fillOpacity": 0.0
                        },
                        highlight_function=lambda x: {"weight": 3, "fillOpacity": 0.2}
                    )
                    popup_content = self.generate_popup_content(properties)
                    gj.add_child(folium.Popup(popup_content, max_width=450))
                    gj.add_child(
                        folium.GeoJsonTooltip(
                            fields=["MSOA"],
                            aliases=["MSOA:"],
                            labels=False
                        )
                    )
                    gj.add_to(region_layer)
                region_layer.add_to(m)
        else:
            msoa_layer = folium.FeatureGroup(name="MSOA Boundaries")
            for _, row in tqdm(
                    self.data_loader.msoa_boundaries.iterrows(),
                    total=self.data_loader.msoa_boundaries.shape[0],
                    desc="Adding MSOA Boundaries",
                    leave=False):
                properties = {
                    "layer_type": "msoa boundary",
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
                        "fillColor": "blue",
                        "color": "red",
                        "weight": 2,
                        "fillOpacity": 0.0
                    },
                    highlight_function=lambda x: {"weight": 3, "fillOpacity": 0.2}
                )
                popup_content = self.generate_popup_content(properties)
                gj.add_child(folium.Popup(popup_content, max_width=450))
                gj.add_child(
                    folium.GeoJsonTooltip(
                        fields=["MSOA"],
                        aliases=["MSOA:"],
                        labels=False
                    )
                )
                gj.add_to(msoa_layer)
            msoa_layer.add_to(m)

    @staticmethod
    def generate_popup_content(properties):
        content = ""
        for key, value in properties.items():
            content += f"<div><strong>{key}:</strong> {value}</div>"
        return content

def get_msoa_map(data_loader, layer_options, center, zoom, selected_region=None, selected_msoa=None):
    mp = MSOAPlotter(data_loader, layer_options=layer_options, center=center, zoom=zoom,
                      selected_region=selected_region, selected_msoa=selected_msoa)
    return mp.map
