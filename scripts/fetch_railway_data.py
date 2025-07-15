import requests
import json
import geopandas as gpd
from pathlib import Path

def fetch_elevation_data(lat, lon):
    """Fetch elevation from OpenElevation API"""
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    response = requests.get(url)
    return response.json()

def fetch_railway_osm(country_bbox):
    """Fetch railway data from OpenStreetMap Overpass API"""
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:25];
    (
      way["railway"~"rail|light_rail|subway|tram"]["railway"!="abandoned"]({country_bbox});
      relation["railway"~"rail|light_rail|subway|tram"]["railway"!="abandoned"]({country_bbox});
    );
    out geom;
    """
    
    response = requests.get(overpass_url, params={'data': overpass_query})
    return response.json()

def fetch_stations_osm(country_bbox):
    """Fetch station data from OpenStreetMap"""
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:25];
    (
      node["railway"="station"]({country_bbox});
      node["public_transport"="station"]["railway"]({country_bbox});
    );
    out;
    """
    
    response = requests.get(overpass_url, params={'data': overpass_query})
    return response.json()