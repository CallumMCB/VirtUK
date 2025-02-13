import folium
from folium.plugins import MarkerCluster
from shapely import total_bounds
from tqdm import tqdm
from shapely.geometry import mapping

from object_identifiers import CareLocations


class MapPlotter:
    def __init__(self, data_loader, layer_options=None, center=None, zoom=None):
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
        self.add_lad_boundaries_layer(m)

        folium.LayerControl(collapsed=False).add_to(m)

    def add_lad_boundaries_layer(self, m):
        if isinstance(self.data_loader.lad_boundaries, dict):
            total_bounds = None  # Will store [minx, miny, maxx, maxy]
            for region, gdf in self.data_loader.lad_boundaries.items():
                region_layer = folium.FeatureGroup(name=f"LAD Boundaries - {region}")
                for _, row in tqdm(gdf.iterrows(),
                                   total=gdf.shape[0],
                                   desc=f"Adding LAD Boundaries for {region}",
                                   leave=False):
                    properties = {
                        "Region": region,
                        "LAD Name": row.get("LAD24NM", "N/A"),
                        "LAD Code": row.get("LAD24CD", "N/A"),
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
                            "weight": 2,
                            "fillOpacity": 0.0
                        }
                    )
                    popup_content = self.generate_popup_content(properties)
                    gj.add_child(folium.Popup(popup_content, max_width=450))
                    gj.add_child(
                        folium.GeoJsonTooltip(
                            fields=["Region", "LAD Name", "LAD Code"],
                            aliases=["Region:", "LAD Name:", "LAD Code:"],
                            labels=False
                        )
                    )
                    gj.add_to(region_layer)

                region_layer.add_to(m)
                # Get the bounds for the current gdf as a list: [minx, miny, maxx, maxy]
                bounds = list(gdf.total_bounds)

                # Initialize total_bounds or update it with the union of bounds
                if total_bounds is None:
                    total_bounds = bounds
                else:
                    total_bounds[0] = min(total_bounds[0], bounds[0])  # minx
                    total_bounds[1] = min(total_bounds[1], bounds[1])  # miny
                    total_bounds[2] = max(total_bounds[2], bounds[2])  # maxx
                    total_bounds[3] = max(total_bounds[3], bounds[3])  # maxy

            # Only adjust map bounds if total_bounds was updated
            if total_bounds is not None:
                m.fit_bounds([[total_bounds[1], total_bounds[0]], [total_bounds[3], total_bounds[2]]])
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

            # Calculate the total bounds of the entire GeoDataFrame:
            # total_bounds returns [minx, miny, maxx, maxy]
            bounds = list(self.data_loader.lad_boundaries.total_bounds)

            # Fit the map to the total bounds
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    @staticmethod
    def generate_popup_content(properties):
        content = ""
        for key, value in properties.items():
            content += f"<div><strong>{key}:</strong> {value}</div>"
        return content

def get_map(data_loader, layer_options, center, zoom):
    mp = MapPlotter(data_loader, layer_options=layer_options, center=center, zoom=zoom)
    return mp.map
