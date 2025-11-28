import math
import random
import time
from typing import Dict, Any

import streamlit as st
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import leafmap.foliumap as leafmap
from streamlit_folium import st_folium


# =========================================================
#  BASIC AUTH USING st.secrets
# =========================================================

def check_credentials(username: str, password: str) -> bool:
    """Check username/password against secrets."""
    try:
        expected_user = st.secrets["auth"]["username"]
        expected_pass = st.secrets["auth"]["password"]
    except Exception:
        st.error("Authentication configuration is missing in secrets.")
        return False

    return (username == expected_user) and (password == expected_pass)


def login_page():
    st.set_page_config(page_title="Wildfire Risk Assessment Dashboard", layout="wide")
    st.title("Wildfire Risk Assessment Dashboard")

    st.markdown("### Login")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("User")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if check_credentials(username, password):
            st.session_state.authenticated = True
            st.session_state.user = username
            st.success("Login successful.")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials. Please try again.")

    st.stop()


# Gate everything behind login
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    login_page()


# =========================================================
#  PREDEFINED AREAS OF INTEREST (AOIs) – “recent searches”
# =========================================================

WILDFIRE_AOIS = {
    "montseny": {
        "name": "Montseny Natural Park",
        "center": {"lat": 41.763979, "lon": 2.407173},
    },
    "gredos": {
        "name": "Sierra de Gredos",
        "center": {"lat": 40.34625, "lon": -5.175306},
    },
    "cazorla": {
        "name": "Sierra de Cazorla",
        "center": {"lat": 37.936667, "lon": -2.958333},
    },
    "cabaneros": {
        "name": "Cabañeros National Park",
        "center": {"lat": 39.396389, "lon": -4.487222},
    },
    "culebra": {
        "name": "Sierra de la Culebra",
        "center": {"lat": 41.78098, "lon": -6.00971},
    },
}


# =========================================================
#  RISK DRIVERS / ACTIONS BY LEVEL
# =========================================================

DRIVERS_BY_LEVEL = {
    "Low": [
        "Discontinuous fuel and mosaic land use.",
        "Predominantly low vegetation height and fuel load.",
        "Good accessibility for suppression resources.",
    ],
    "Mid": [
        "Continuous shrubland and forest patches within the ROI.",
        "Moderate slopes increasing potential fire spread rate.",
        "Wildland–urban interface within or near ROI.",
    ],
    "High": [
        "Dense, continuous forest or shrubland with high fuel loads.",
        "Complex topography with strong slope exposure.",
        "Limited access roads and constrained suppression logistics.",
    ],
}

ACTIONS_BY_LEVEL = {
    "Low": [
        "Maintain baseline monitoring and routine patrols.",
        "Keep local communities informed about general fire safety.",
        "Preserve current fuel management practices.",
    ],
    "Mid": [
        "Increase monitoring frequency during critical weather windows.",
        "Pre-position suppression resources near WUI sectors.",
        "Plan and schedule preventive fuel reduction treatments.",
    ],
    "High": [
        "Activate reinforced monitoring and early detection protocols.",
        "Prepare and test evacuation and communication plans.",
        "Coordinate proactively with regional and national emergency services.",
    ],
}


# =========================================================
#  CONFIG, GEOCODER, RISK "DB"
# =========================================================

st.set_page_config(
    page_title="Wildfire Risk Assessment Dashboard",
    layout="wide",
)

geolocator = Nominatim(user_agent="wildfire_dashboard")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


def simulate_loading(label: str = "Running wildfire risk engine...", duration_s: float = 6.0):
    """Simulated loading bar to make the app feel like it's processing."""
    progress = st.progress(0)
    status = st.empty()
    status.text(label)
    steps = 100
    sleep_time = duration_s / steps
    for i in range(steps):
        time.sleep(sleep_time)
        progress.progress(i + 1)
    status.empty()
    progress.empty()


def query_risk_from_db(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    scenario: str,
) -> Dict[str, Any]:
    """
    Mock: risk level as a function of ROI radius only.
    Replace with real DB / model inference later.
    """
    if radius_km <= 10:
        risk_index = 0.35
        risk_level = "Low"
    elif radius_km <= 20:
        risk_index = 0.55
        risk_level = "Mid"
    else:  # 20–30 km
        risk_index = 0.80
        risk_level = "High"

    drivers = DRIVERS_BY_LEVEL[risk_level]
    recommended_actions = ACTIONS_BY_LEVEL[risk_level]

    sat_image_path = "data/mock_sat_image.png"

    return {
        "risk_index": risk_index,
        "risk_level": risk_level,
        "drivers": drivers,
        "recommended_actions": recommended_actions,
        "sat_image": sat_image_path,
    }


# =========================================================
#  GEOMETRY / STYLE HELPERS
# =========================================================

def build_circle_polygon(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    num_points: int = 64,
):
    """Approximate circular polygon around center with given radius (km)."""
    lat_radius_deg = radius_km / 111.0
    lon_radius_deg = radius_km / (111.0 * math.cos(math.radians(center_lat)))

    coords = []
    for i in range(num_points + 1):
        angle = 2 * math.pi * i / num_points
        dlat = lat_radius_deg * math.sin(angle)
        dlon = lon_radius_deg * math.cos(angle)
        lat = center_lat + dlat
        lon = center_lon + dlon
        coords.append([lon, lat])

    return coords


def build_roi_geojson(center_lat: float, center_lon: float, radius_km: float, risk_level: str):
    polygon = build_circle_polygon(center_lat, center_lon, radius_km)
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Selected ROI",
                    "risk_level": risk_level,
                    "radius_km": radius_km,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon],
                },
            }
        ],
    }
    return geojson


def build_irregular_blob_polygon(
    center_lat: float,
    center_lon: float,
    base_radius_km: float,
    jitter_km: float,
    num_vertices: int,
    rng: random.Random,
):
    """
    Irregular polygon around a local centroid with smooth contour (no star-shape).

    - Angles: uniformly spaced (no angular jitter).
    - Radii: random around base_radius_km, then smoothed with neighbour averaging.
    """
    # 1) Generamos radios con ruido
    raw_radii = [
        base_radius_km + rng.uniform(-jitter_km, jitter_km)
        for _ in range(num_vertices)
    ]

    # 2) Suavizado simple (media de vecinos). Dos pasadas para más suavidad.
    def smooth(radii):
        n = len(radii)
        smoothed = []
        for i in range(n):
            r_prev = radii[(i - 1) % n]
            r_curr = radii[i]
            r_next = radii[(i + 1) % n]
            smoothed.append((r_prev + r_curr + r_next) / 3.0)
        return smoothed

    radii = smooth(raw_radii)
    radii = smooth(radii)  # segunda pasada de suavizado

    coords = []
    lat_center_rad = math.radians(center_lat)

    for i in range(num_vertices):
        # ángulos uniformes
        theta = 2.0 * math.pi * i / num_vertices

        r_km = max(radii[i], 0.5)  # evitamos radios demasiado pequeños

        lat_radius_deg = r_km / 111.0
        lon_radius_deg = r_km / (111.0 * math.cos(lat_center_rad))

        dlat = lat_radius_deg * math.sin(theta)
        dlon = lon_radius_deg * math.cos(theta)

        lat = center_lat + dlat
        lon = center_lon + dlon
        coords.append([lon, lat])

    # cerrar polígono
    coords.append(coords[0])
    return coords


def build_subzones_geojson(region_key: str, max_radius_km: float = 30.0) -> dict:
    """
    Non-overlapping irregular sub-zones around AOI center.
    """
    region = WILDFIRE_AOIS[region_key]
    center = region["center"]

    rng = random.Random(region_key)  # deterministic per AOI

    center_lat = center["lat"]
    center_lon = center["lon"]
    center_lat_rad = math.radians(center_lat)

    n_subzones = rng.randint(3, 6)
    base_spacing_deg = 360.0 / n_subzones
    global_angle_offset = rng.uniform(0.0, base_spacing_deg)

    features = []

    for i in range(n_subzones):
        risk_level = rng.choice(["Low", "Mid", "High"])

        bearing_deg = global_angle_offset + i * base_spacing_deg + rng.uniform(-10.0, 10.0)
        bearing = math.radians(bearing_deg)

        dist_centroid_km = rng.uniform(8.0, 15.0)
        dist_centroid_km = min(dist_centroid_km, max_radius_km - 5.0)

        dlat_centroid_deg = (dist_centroid_km / 111.0) * math.sin(bearing)
        dlon_centroid_deg = (dist_centroid_km / (111.0 * math.cos(center_lat_rad))) * math.cos(
            bearing
        )

        centroid_lat = center_lat + dlat_centroid_deg
        centroid_lon = center_lon + dlon_centroid_deg

        base_radius_km = rng.uniform(2.0, 4.0)
        jitter_km = rng.uniform(0.3, 0.7)  # menos variación → contorno más suave
        num_vertices = rng.randint(30, 45)  # muchas caras, pero suavizadas

        polygon = build_irregular_blob_polygon(
            center_lat=centroid_lat,
            center_lon=centroid_lon,
            base_radius_km=base_radius_km,
            jitter_km=jitter_km,
            num_vertices=num_vertices,
            rng=rng,
        )

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": f"{region_key}_{i+1}",
                    "name": f"{region['name']} – zone {i+1}",
                    "risk_level": risk_level,
                    "region": region["name"],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon],
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


def roi_style_function(feature):
    """ROI: outline only, no fill."""
    return {
        "color": "#0000ff",
        "weight": 2,
        "fillColor": "#000000",
        "fillOpacity": 0.0,
    }


def risk_style_function(feature):
    """Subzones: colour by risk_level."""
    level = feature["properties"].get("risk_level", "Low")

    if level == "High":
        color = "#dc143c"  # red
        fill_opacity = 0.40
    elif level in ("Moderate", "Mid"):
        color = "#ffa500"  # orange
        fill_opacity = 0.35
    else:
        color = "#228b22"  # green
        fill_opacity = 0.30

    return {
        "color": color,
        "weight": 2,
        "fillColor": color,
        "fillOpacity": fill_opacity,
    }


def estimate_zoom(radius_km: float) -> int:
    if radius_km <= 10:
        return 11
    elif radius_km <= 20:
        return 10
    else:  # 20–30
        return 9


# =========================================================
#  MAIN APP (user already authenticated)
# =========================================================

st.title("Wildfire Risk Assessment Dashboard")

st.sidebar.header("Region of Interest")

# ROI radius (5–30 km)
radius_km = st.sidebar.slider(
    "ROI radius [km]",
    min_value=5,
    max_value=30,
    value=15,
    step=5,
)

scenario = st.sidebar.selectbox(
    "Scenario",
    options=["Current conditions", "48h forecast", "7-day outlook"],
)

# Basemap selection
basemap_option = st.sidebar.selectbox(
    "Basemap",
    options=[
        "CartoDB.Voyager (streets)",
        "Esri.WorldImagery (satellite)",
    ],
)

# Session state defaults for current ROI center (use Montseny as non-urban default)
if "center_lat" not in st.session_state:
    st.session_state.center_lat = WILDFIRE_AOIS["montseny"]["center"]["lat"]
if "center_lon" not in st.session_state:
    st.session_state.center_lon = WILDFIRE_AOIS["montseny"]["center"]["lon"]
if "selected_aoi_key" not in st.session_state:
    st.session_state.selected_aoi_key = "montseny"  # default AOI

# ----------------------------------
# Free ROI search (geocoding)
# ----------------------------------
st.sidebar.subheader("Search location")
search_query = st.sidebar.text_input(
    "Location (city, region, address)",
    value="Montseny Natural Park",
)
search_button = st.sidebar.button("Update ROI from search")

st.sidebar.caption(
    "Location search powered by OpenStreetMap (Nominatim). "
    "In production, this would be replaced by the project's geocoding service."
)

if search_button and search_query.strip():
    location = geocode(search_query)
    if location is not None:
        simulate_loading("Running wildfire risk engine for new ROI...")
        st.session_state.center_lat = location.latitude
        st.session_state.center_lon = location.longitude
        st.session_state.selected_aoi_key = None
    else:
        st.sidebar.error("Location not found. Please refine your search.")

# ----------------------------------
# "Recent AOIs" – looks like history / saved searches
# ----------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Recent AOIs")

aoi_keys = list(WILDFIRE_AOIS.keys())
aoi_options = ["(none)"] + aoi_keys

old_aoi_key = st.session_state.selected_aoi_key

selected_aoi_option = st.sidebar.selectbox(
    "Select AOI",
    options=aoi_options,
    format_func=lambda k: "(none)" if k == "(none)" else WILDFIRE_AOIS[k]["name"],
)

new_aoi_key = None if selected_aoi_option == "(none)" else selected_aoi_option

if new_aoi_key != old_aoi_key:
    st.session_state.selected_aoi_key = new_aoi_key
    if new_aoi_key is not None:
        aoi = WILDFIRE_AOIS[new_aoi_key]
        simulate_loading(f"Running wildfire risk engine for {aoi['name']}...")
        center = aoi["center"]
        st.session_state.center_lat = center["lat"]
        st.session_state.center_lon = center["lon"]

center_lat = st.session_state.center_lat
center_lon = st.session_state.center_lon

# ---------------------------------------------------------
#  Risk query for current ROI
# ---------------------------------------------------------
risk_result = query_risk_from_db(
    center_lat=center_lat,
    center_lon=center_lon,
    radius_km=radius_km,
    scenario=scenario,
)

geojson_roi = build_roi_geojson(
    center_lat=center_lat,
    center_lon=center_lon,
    radius_km=radius_km,
    risk_level=risk_result["risk_level"],
)

zoom = estimate_zoom(radius_km)

# =========================================================
#  LAYOUT: MAP + INFO PANEL
# =========================================================

map_col, info_col = st.columns((2.2, 1.8))

with map_col:
    st.subheader("Map & overlays")

    m = leafmap.Map(
        center=[center_lat, center_lon],
        zoom=zoom,
        Draw_export=False,
        locate_control=False,
    )

    # Basemap selection logic
    if "CartoDB.Voyager" in basemap_option:
        m.add_basemap("CartoDB.Voyager")
    else:
        m.add_basemap("Esri.WorldImagery")

    # ROI outline
    m.add_geojson(
        geojson_roi,
        layer_name="ROI",
        style_function=roi_style_function,
    )

    # Risk patches only for AOIs
    subzones_geojson = None
    if st.session_state.selected_aoi_key is not None:
        subzones_geojson = build_subzones_geojson(
            st.session_state.selected_aoi_key, max_radius_km=30.0
        )
        m.add_geojson(
            subzones_geojson,
            layer_name="Risk subzones",
            style_function=risk_style_function,
        )

    st_folium(m, width="100%", height=600)

    if st.session_state.selected_aoi_key is not None:
        name = WILDFIRE_AOIS[st.session_state.selected_aoi_key]["name"]
        st.caption(
            f"ROI centred on {name} (radius {radius_km} km). "
            "Coloured patches represent areas where the risk engine has identified elevated wildfire risk."
        )
    else:
        st.caption(
            f"ROI centred at the selected location (radius {radius_km} km). "
            "Use the search bar or pick a recent AOI to explore different areas."
        )

    st.subheader("Satellite imagery")
    if risk_result["sat_image"]:
        caption_location = (
            WILDFIRE_AOIS[st.session_state.selected_aoi_key]["name"]
            if st.session_state.selected_aoi_key is not None
            else "selected ROI"
        )
        st.image(
            risk_result["sat_image"],
            caption=f"Representative satellite view for {caption_location} ({scenario})",
            use_column_width=True,
        )
    else:
        st.info("No satellite imagery available for this ROI.")


with info_col:
    st.subheader("Risk assessment")

    st.metric(
        label="Risk Index (0–1)",
        value=f"{risk_result['risk_index']:.2f}",
    )
    st.metric(
        label="Risk Level (ROI)",
        value=risk_result["risk_level"],
    )

    if st.session_state.selected_aoi_key is not None and subzones_geojson is not None:
        st.markdown("**Risk subzones in current AOI**")

        present_levels = set()
        for feat in subzones_geojson["features"]:
            name = feat["properties"]["name"]
            level = feat["properties"]["risk_level"]
            present_levels.add(level)
            st.markdown(f"- **{name}** → {level}")

        order = ["High", "Mid", "Low"]
        present_levels_ordered = [lvl for lvl in order if lvl in present_levels]

        with st.expander("Main drivers by risk level", expanded=False):
            for lvl in present_levels_ordered:
                st.markdown(f"**{lvl} risk**")
                for d in DRIVERS_BY_LEVEL[lvl]:
                    st.markdown(f"- {d}")

        with st.expander("Recommended actions by risk level", expanded=False):
            for lvl in present_levels_ordered:
                st.markdown(f"**{lvl} risk**")
                for a in ACTIONS_BY_LEVEL[lvl]:
                    st.markdown(f"- {a}")
    else:
        with st.expander("Main drivers", expanded=False):
            for d in risk_result["drivers"]:
                st.markdown(f"- {d}")

        with st.expander("Recommended actions", expanded=False):
            for a in risk_result["recommended_actions"]:
                st.markdown(f"- {a}")

    st.markdown("---")
    st.caption(
        "All risk indicators, subzones and recommendations in this prototype are illustrative. "
        "In an operational system they would be generated from the underlying AI risk engine "
        "and geospatial data pipelines."
    )
