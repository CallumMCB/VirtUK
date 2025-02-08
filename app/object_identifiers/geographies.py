import folium
from tqdm import tqdm
from shapely.geometry import mapping

class Geographies:
    def __init__(self, map_plotter):
        self.mp = map_plotter
        self.data_loader = self.mp.data_loader

    def add_msoa_boundaries_layer(self, m):
        if isinstance(self.data_loader.msoa_boundaries, dict):
            for region, gdf in self.data_loader.msoa_boundaries.items():
                msoa_layer = folium.FeatureGroup(name=f"MSOA Boundaries - {region}")
                for _, row in tqdm(
                    gdf.iterrows(),
                    total=gdf.shape[0],
                    desc=f"Adding MSOA Boundaries for {region}",
                    leave=False
                ):
                    properties = {
                        "layer_type": "boundary",
                        "MSOA": row["MSOA21CD"],
                        "Care homes with nursing": row["Care homes with nursing"],
                        "Care homes without nursing": row["Care homes without nursing"],
                        "Total Care Home Residents": row["Total Care Homes"],
                        "Male Residents 65+": row["Male 65+"],
                        "Female Residents 65+": row["Female 65+"],
                        "Staff/Owners": row["staff/owner"]
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
                    popup_content = self.mp.generate_popup_content(properties)
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
        else:
            msoa_layer = folium.FeatureGroup(name="MSOA Boundaries")
            for _, row in tqdm(
                self.data_loader.msoa_boundaries.iterrows(),
                total=self.data_loader.msoa_boundaries.shape[0],
                desc="Adding MSOA Boundaries",
                leave=False
            ):
                properties = {
                    "layer_type": "boundary",
                    "MSOA": row["MSOA21CD"],
                    "Care homes with nursing": row["Care homes with nursing"],
                    "Care homes without nursing": row["Care homes without nursing"],
                    "Total Care Home Residents": row["Total Care Homes"],
                    "Male Residents 65+": row["Male 65+"],
                    "Female Residents 65+": row["Female 65+"],
                    "Staff/Owners": row["staff/owner"]
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
                popup_content = self.mp.generate_popup_content(properties)
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
                lad_layer = folium.FeatureGroup(name=f"LAD Boundaries - {region}")
                for _, row in tqdm(
                    gdf.iterrows(),
                    total=gdf.shape[0],
                    desc=f"Adding LAD Boundaries for {region}",
                    leave=False
                ):
                    properties = {
                        "layer_type": "boundary",
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
                            "weight": 1,
                            "fillOpacity": 0.0
                        }
                    )
                    gj.add_child(folium.Tooltip(f"LAD: {row.get('LAD_NAME', 'N/A')}"))
                    gj.add_to(lad_layer)
                lad_layer.add_to(m)
        else:
            lad_layer = folium.FeatureGroup(name="LAD Boundaries")
            for _, row in tqdm(
                self.data_loader.lad_boundaries.iterrows(),
                total=self.data_loader.lad_boundaries.shape[0],
                desc="Adding LAD Boundaries",
                leave=False
            ):
                properties = {
                    "layer_type": "boundary",
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
                        "weight": 1,
                        "fillOpacity": 0.0
                    }
                )
                gj.add_child(folium.Tooltip(f"LAD: {row.get('LAD_NAME', 'N/A')}"))
                gj.add_to(lad_layer)
            lad_layer.add_to(m)
