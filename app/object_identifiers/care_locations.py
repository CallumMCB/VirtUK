import folium
from folium.plugins import MarkerCluster
from tqdm import tqdm

class CareLocations:
    def __init__(self, map_plotter):
        self.mp = map_plotter
        self.data_loader = self.mp.data_loader

    def add_care_homes(self, m):
        # Check if care homes data is stored by region.
        if isinstance(self.data_loader.df_care_homes, dict):
            # Loop through each region.
            for region, df in self.data_loader.df_care_homes.items():
                care_layer = folium.FeatureGroup(name=f"Care Homes - {region}")
                cluster = MarkerCluster(options={'disableClusteringAtZoom': 10}).add_to(care_layer)
                for _, row in tqdm(
                    df.iterrows(),
                    total=df.shape[0],
                    desc=f"Adding Care Homes for {region}",
                    leave=False
                ):
                    properties = {
                        "layer_type": "feature",
                        "feature_type": "care_home",
                        "name": row["Location Name"],
                        "postcode": row["Location Postal Code"],
                    }
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [row["Location Longitude"], row["Location Latitude"]]
                        },
                        "properties": properties
                    }
                    popup_content = self.mp.generate_popup_content(properties)
                    gj = folium.GeoJson(data=feature)
                    gj.add_child(folium.Popup(popup_content, max_width=450))
                    gj.add_to(cluster)
                care_layer.add_to(m)
        else:
            # Single dataset mode.
            care_layer = folium.FeatureGroup(name="Care Homes")
            cluster = MarkerCluster(options={'disableClusteringAtZoom': 10}).add_to(care_layer)
            for _, row in tqdm(
                self.data_loader.df_care_homes.iterrows(),
                total=self.data_loader.df_care_homes.shape[0],
                desc="Adding Care Homes",
                leave=False
            ):
                properties = {
                    "layer_type": "feature",
                    "feature_type": "care_home",
                    "name": row["Location Name"],
                    "postcode": row["Location Postal Code"],
                }
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [row["Location Longitude"], row["Location Latitude"]]
                    },
                    "properties": properties
                }
                popup_content = self.mp.generate_popup_content(properties)
                gj = folium.GeoJson(data=feature)
                gj.add_child(folium.Popup(popup_content, max_width=450))
                gj.add_to(cluster)
            care_layer.add_to(m)

    def add_hospices(self, m):
        if isinstance(self.data_loader.df_hospices, dict):
            for region, df in self.data_loader.df_hospices.items():
                hospice_layer = folium.FeatureGroup(name=f"Hospices - {region}")
                cluster = MarkerCluster(options={'disableClusteringAtZoom': 10}).add_to(hospice_layer)
                for _, row in tqdm(
                    df.iterrows(),
                    total=df.shape[0],
                    desc=f"Adding Hospices for {region}",
                    leave=False
                ):
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [row["longitude"], row["latitude"]]
                        },
                        "properties": {
                            "layer_type": "feature",
                            "feature_type": "hospice",
                            "name": row["Name"],
                            "trust": row["Is Operated By - Code"]
                        }
                    }
                    popup_content = self.mp.generate_popup_content(feature["properties"])
                    gj = folium.GeoJson(data=feature)
                    gj.add_child(folium.Popup(popup_content, max_width=450))
                    gj.add_to(cluster)
                hospice_layer.add_to(m)
        else:
            hospice_layer = folium.FeatureGroup(name="Hospices")
            cluster = MarkerCluster(options={'disableClusteringAtZoom': 10}).add_to(hospice_layer)
            for _, row in tqdm(
                self.data_loader.df_hospices.iterrows(),
                total=self.data_loader.df_hospices.shape[0],
                desc="Adding Hospices",
                leave=False
            ):
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [row["longitude"], row["latitude"]]
                    },
                    "properties": {
                        "layer_type": "feature",
                        "feature_type": "hospice",
                        "name": row["Name"],
                        "trust": row["Is Operated By - Code"]
                    }
                }
                popup_content = self.mp.generate_popup_content(feature["properties"])
                gj = folium.GeoJson(data=feature)
                gj.add_child(folium.Popup(popup_content, max_width=450))
                gj.add_to(cluster)
            hospice_layer.add_to(m)

    def add_hospitals(self, m):
        if isinstance(self.data_loader.df_hospitals, dict):
            for region, df in self.data_loader.df_hospitals.items():
                hospital_layer = folium.FeatureGroup(name=f"Hospitals - {region}")
                cluster = MarkerCluster(options={'disableClusteringAtZoom': 10}).add_to(hospital_layer)
                for _, row in tqdm(
                    df.iterrows(),
                    total=df.shape[0],
                    desc=f"Adding Hospitals for {region}",
                    leave=False
                ):
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [row["longitude"], row["latitude"]]
                        },
                        "properties": {
                            "layer_type": "feature",
                            "feature_type": "hospital",
                            "name": row["Name"],
                            "trust": row["Is Operated By - Code"]
                        }
                    }
                    popup_content = self.mp.generate_popup_content(feature["properties"])
                    gj = folium.GeoJson(data=feature)
                    gj.add_child(folium.Popup(popup_content, max_width=450))
                    gj.add_to(cluster)
                hospital_layer.add_to(m)
        else:
            hospital_layer = folium.FeatureGroup(name="Hospitals")
            cluster = MarkerCluster(options={'disableClusteringAtZoom': 10}).add_to(hospital_layer)
            for _, row in tqdm(
                self.data_loader.df_hospitals.iterrows(),
                total=self.data_loader.df_hospitals.shape[0],
                desc="Adding Hospitals",
                leave=False
            ):
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [row["longitude"], row["latitude"]]
                    },
                    "properties": {
                        "layer_type": "feature",
                        "feature_type": "hospital",
                        "name": row["Name"],
                        "trust": row["Is Operated By - Code"]
                    }
                }
                popup_content = self.mp.generate_popup_content(feature["properties"])
                gj = folium.GeoJson(data=feature)
                gj.add_child(folium.Popup(popup_content, max_width=450))
                gj.add_to(cluster)
            hospital_layer.add_to(m)

    def add_trusts(self, m):
        if isinstance(self.data_loader.df_trusts, dict):
            for region, df in self.data_loader.df_trusts.items():
                trust_layer = folium.FeatureGroup(name=f"NHS Trusts - {region}")
                cluster = MarkerCluster().add_to(trust_layer)
                for _, row in tqdm(
                    df.iterrows(),
                    total=df.shape[0],
                    desc=f"Adding NHS Trusts for {region}",
                    leave=False
                ):
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [row["longitude"], row["latitude"]]
                        },
                        "properties": {
                            "layer_type": "feature",
                            "feature_type": "trust",
                            "name": row["org_name"]
                        }
                    }
                    popup_content = self.mp.generate_popup_content(feature["properties"])
                    gj = folium.GeoJson(data=feature)
                    gj.add_child(folium.Popup(popup_content, max_width=250))
                    gj.add_to(cluster)
                trust_layer.add_to(m)
        else:
            trust_layer = folium.FeatureGroup(name="NHS Trusts")
            cluster = MarkerCluster().add_to(trust_layer)
            for _, row in tqdm(
                self.data_loader.df_trusts.iterrows(),
                total=self.data_loader.df_trusts.shape[0],
                desc="Adding NHS Trusts",
                leave=False
            ):
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [row["longitude"], row["latitude"]]
                    },
                    "properties": {
                        "layer_type": "feature",
                        "feature_type": "trust",
                        "name": row["org_name"]
                    }
                }
                popup_content = self.mp.generate_popup_content(feature["properties"])
                gj = folium.GeoJson(data=feature)
                gj.add_child(folium.Popup(popup_content, max_width=250))
                gj.add_to(cluster)
            trust_layer.add_to(m)
