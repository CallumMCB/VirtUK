import folium
from folium.plugins import MarkerCluster
from tqdm import tqdm
from shapely.geometry import mapping

# (Assuming CareLocations is defined elsewhere in your project.)
from object_identifiers import CareLocations


class MapPlotter:
    def __init__(self, data_loader, layer_options=None, center=None, zoom=None):
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
        self.create_map()

    def create_map(self):
        m = self.map
        # Feature layers.
        if self.layer_options.get("hospitals", False):
            self.cloc.add_hospitals(m)
        if self.layer_options.get("hospices", False):
            self.cloc.add_hospices(m)
        if self.layer_options.get("trusts", False):
            self.cloc.add_trusts(m)
        if self.layer_options.get("care_homes", False):
            self.cloc.add_care_homes(m)
        # Boundary layers.
        if self.layer_options.get("lad_boundaries", False):
            self.add_lad_boundaries_layer(m)
        if self.layer_options.get("msoa_boundaries", True):
            self.add_msoa_boundaries_layer(m)

        folium.LayerControl(collapsed=False).add_to(m)

    def add_msoa_boundaries_layer(self, m):
        # Check if msoa_boundaries is a dict (i.e. multi-region mode).
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
                        "MSOA name": row["MSOA21NM"],
                        "Region": region
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
                            fields=["Region", "MSOA name", "MSOA"],
                            aliases=["Region:", "MSOA name:", "MSOA:"],
                            labels=False
                        )
                    )
                    gj.add_to(region_layer)
                region_layer.add_to(m)
        else:
            # Single dataset mode.
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

    def add_lad_boundaries_layer(self, m):
        if isinstance(self.data_loader.lad_boundaries, dict):
            for region, gdf in self.data_loader.lad_boundaries.items():
                region_layer = folium.FeatureGroup(name=f"LAD Boundaries - {region}")
                for _, row in tqdm(gdf.iterrows(),
                                   total=gdf.shape[0],
                                   desc=f"Adding LAD Boundaries for {region}",
                                   leave=False):
                    properties = {
                        "layer_type": "lad boundary",
                        "LAD": row.get("LAD_NAME", "N/A"),
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
                            "fillColor": "gray",
                            "color": "black",
                            "weight": 4,
                            "fillOpacity": 0.0
                        }
                    )
                    tooltip_text = f"LAD: {row.get('LAD24NM', 'N/A')}"
                    gj.add_child(folium.Tooltip(tooltip_text))
                    gj.add_to(region_layer)
                region_layer.add_to(m)
        else:
            lad_layer = folium.FeatureGroup(name="LAD Boundaries")
            for _, row in tqdm(
                    self.data_loader.lad_boundaries.iterrows(),
                    total=self.data_loader.lad_boundaries.shape[0],
                    desc="Adding LAD Boundaries",
                    leave=False):
                properties = {
                    "layer_type": "lad boundary",
                    "LAD": row.get("LAD_NAME", "N/A")
                }
                feature = {
                    "type": "Feature",
                    "geometry": mapping(row["geometry"]),
                    "properties": properties
                }
                gj = folium.GeoJson(
                    data=feature,
                    style_function=lambda x: {
                        "fillColor": "gray",
                        "color": "black",
                        "weight": 4,
                        "fillOpacity": 0.0
                    }
                )
                gj.add_child(folium.Tooltip(f"LAD: {row.get('LAD24NM', 'N/A')}"))
                gj.add_to(lad_layer)
            lad_layer.add_to(m)

    @staticmethod
    def generate_popup_content(properties):
        content = ""
        for key, value in properties.items():
            content += f"<div><strong>{key}:</strong> {value}</div>"
        return content

def get_map(data_loader, layer_options, center, zoom):
    mp = MapPlotter(data_loader, layer_options=layer_options, center=center, zoom=zoom)
    return mp.map
