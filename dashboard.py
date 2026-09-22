import streamlit as st
import pandas as pd
import requests
import math
import folium
from streamlit_folium import st_folium

# ============================================================
# FINAL COUNTER-UAV DASHBOARD
# Public Internet UAV Replay + Satellite Maps + Tracking Demo
# Waiting only for real YOLO11s / BoT-SORT outputs to replace
# the temporary Internet/Demo sources.
# ============================================================

DATASET_NAME = "riotu-lab/os-rfodg-outdoor-uav-synthetic-dataset-taif-saudi-arabia"

st.set_page_config(
    page_title="Counter-UAV Monitoring Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROFESSIONAL DARK UI
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #08111f;
            color: #e8eef7;
        }

        [data-testid="stSidebar"] {
            background: #0d1726;
            border-right: 1px solid #1e2d43;
        }

        h1, h2, h3 {
            color: #f3f7fb !important;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1800px;
        }

        [data-testid="stMetric"] {
            background: #0f1b2d;
            border: 1px solid #21344d;
            border-radius: 14px;
            padding: 14px 16px;
        }

        [data-testid="stMetricLabel"] {
            color: #9fb1c8;
        }

        [data-testid="stMetricValue"] {
            color: #ffffff;
        }

        .section-card {
            background: #0f1b2d;
            border: 1px solid #21344d;
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }

        .small-note {
            color: #92a5bd;
            font-size: 0.85rem;
        }

        .status-chip {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid #2b4160;
            background: #12243a;
            margin-right: 6px;
            font-size: 0.85rem;
        }

        .main-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }

        .main-subtitle {
            color: #9fb1c8;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">COUNTER-UAV MONITORING & DECISION DASHBOARD</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-subtitle">'
    'Internet UAV telemetry • Satellite monitoring • Tracking status • Alerts • Human-in-the-loop safety decision'
    '</div>',
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(ttl=600)
def load_uav_data():
    split_response = requests.get(
        "https://datasets-server.huggingface.co/splits",
        params={"dataset": DATASET_NAME},
        timeout=30,
    )
    split_response.raise_for_status()
    split_json = split_response.json()

    config_name = split_json["splits"][0]["config"]
    split_name = split_json["splits"][0]["split"]

    rows_response = requests.get(
        "https://datasets-server.huggingface.co/rows",
        params={
            "dataset": DATASET_NAME,
            "config": config_name,
            "split": split_name,
            "offset": 0,
            "length": 100,
        },
        timeout=30,
    )
    rows_response.raise_for_status()
    rows_json = rows_response.json()

    df = pd.DataFrame([item["row"] for item in rows_json["rows"]])

    numeric_columns = [
        "timestamp", "vel_x", "vel_y", "vel_z",
        "latitude", "longitude", "altitude",
        "lidar_range", "AMSL"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["latitude", "longitude", "vel_x", "vel_y", "vel_z"]
    ).reset_index(drop=True)

    df["speed"] = (
        df["vel_x"] ** 2
        + df["vel_y"] ** 2
        + df["vel_z"] ** 2
    ) ** 0.5

    return df


@st.cache_data(ttl=86400)
def load_saudi_boundary():
    url = (
        "https://raw.githubusercontent.com/"
        "nvkelso/natural-earth-vector/master/"
        "geojson/ne_110m_admin_0_countries.geojson"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    countries = response.json()

    for feature in countries["features"]:
        properties = feature.get("properties", {})
        if (
            properties.get("ADMIN") == "Saudi Arabia"
            or properties.get("NAME") == "Saudi Arabia"
            or properties.get("ADM0_A3") == "SAU"
            or properties.get("ISO_A3") == "SAU"
        ):
            return {
                "type": "FeatureCollection",
                "features": [feature],
            }

    return None


def get_direction(vx, vy):
    angle = (math.degrees(math.atan2(vy, vx)) + 360) % 360
    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    index = int((angle + 22.5) // 45) % 8
    return directions[index], angle


def calculate_distance(lat1, lon1, lat2, lon2):
    earth_radius = 6371000

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * math.asin(math.sqrt(a))


# ============================================================
# LOAD INTERNET DATA
# ============================================================

try:
    data = load_uav_data()
except Exception as error:
    st.error("Could not load the public Internet UAV dataset.")
    st.code(str(error))
    st.stop()

if data.empty:
    st.error("No valid UAV telemetry was received.")
    st.stop()

try:
    saudi_boundary = load_saudi_boundary()
except Exception:
    saudi_boundary = None


# ============================================================
# SESSION STATE
# ============================================================

def ensure_state(name, default):
    if name not in st.session_state:
        st.session_state[name] = default


ensure_state("sample_index", 0)
ensure_state("playing", False)
ensure_state("tracking_state", "TRACKING")
ensure_state("reacq_attempts", 0)
ensure_state("safe_state", False)
ensure_state("safe_rejected", False)
ensure_state("auto_follow", True)
ensure_state("demo_reacq_result", "Success")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("CONTROL PANEL")

control_col1, control_col2 = st.sidebar.columns(2)

with control_col1:
    if st.button("▶ Start", use_container_width=True):
        st.session_state.playing = True

with control_col2:
    if st.button("⏸ Pause", use_container_width=True):
        st.session_state.playing = False

if st.sidebar.button("↻ Restart Replay", use_container_width=True):
    st.session_state.sample_index = 0
    st.session_state.playing = False
    st.rerun()

st.sidebar.toggle("Auto-follow UAV", key="auto_follow")

st.sidebar.divider()

st.sidebar.subheader("Internet Data")
st.sidebar.success("CONNECTED")
st.sidebar.caption("Public synthetic UAV telemetry")
st.sidebar.write("Region: Taif, Saudi Arabia")
st.sidebar.write("Replay interval: 2 seconds")

st.sidebar.divider()

st.sidebar.subheader("Tracking Demo")
st.sidebar.caption(
    "Temporary simulation until real YOLO11s + BoT-SORT outputs arrive."
)

st.sidebar.selectbox(
    "Re-acquisition Result",
    ["Success", "Failure"],
    key="demo_reacq_result",
)

if st.sidebar.button("⚠ Simulate Target LOST", use_container_width=True):
    st.session_state.tracking_state = "LOST"
    st.session_state.reacq_attempts = 0
    st.session_state.safe_state = False
    st.session_state.safe_rejected = False
    st.rerun()

if st.sidebar.button("✓ Restore Tracking", use_container_width=True):
    st.session_state.tracking_state = "TRACKING"
    st.session_state.reacq_attempts = 0
    st.session_state.safe_state = False
    st.session_state.safe_rejected = False
    st.rerun()


# ============================================================
# MAIN DYNAMIC DASHBOARD
# ============================================================

run_every = 2 if (
    st.session_state.playing
    or st.session_state.tracking_state in ["LOST", "RE-ACQUIRING"]
) else None


@st.fragment(run_every=run_every)
def dashboard_body():

    # --------------------------------------------------------
    # REPLAY
    # --------------------------------------------------------
    if st.session_state.playing:
        st.session_state.sample_index = (
            st.session_state.sample_index + 1
        ) % len(data)

    current_index = min(st.session_state.sample_index, len(data) - 1)
    row = data.iloc[current_index]

    latitude = float(row["latitude"])
    longitude = float(row["longitude"])

    direction, heading = get_direction(
        float(row["vel_x"]),
        float(row["vel_y"]),
    )

    # --------------------------------------------------------
    # PROTECTED ZONE
    # --------------------------------------------------------
    zone_index = min(70, len(data) - 1)

    zone_lat = float(data.iloc[zone_index]["latitude"])
    zone_lon = float(data.iloc[zone_index]["longitude"])

    distance_to_zone = calculate_distance(
        latitude,
        longitude,
        zone_lat,
        zone_lon,
    )

    # --------------------------------------------------------
    # THREAT ASSESSMENT
    # --------------------------------------------------------
    if distance_to_zone <= 12:
        threat_level = "HIGH"
        recommendation = "Alert Operator"

    elif distance_to_zone <= 30:
        threat_level = "MEDIUM"
        recommendation = "Increase Monitoring"

    else:
        threat_level = "LOW"
        recommendation = "Continue Monitoring"

    # --------------------------------------------------------
    # DEMO AUTOMATIC RE-ACQUISITION
    # --------------------------------------------------------
    if st.session_state.tracking_state == "LOST":
        st.session_state.tracking_state = "RE-ACQUIRING"
        st.session_state.reacq_attempts = 0

    elif st.session_state.tracking_state == "RE-ACQUIRING":
        st.session_state.reacq_attempts += 1

        if st.session_state.reacq_attempts >= 2:

            if st.session_state.demo_reacq_result == "Success":
                st.session_state.tracking_state = "TRACKING"
                st.session_state.reacq_attempts = 0

            else:
                st.session_state.tracking_state = "FAILED"

    # ========================================================
    # TOP METRICS
    # ========================================================

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    m1.metric("UAV ID", "TAIF-01")
    m2.metric("Speed", f"{row['speed']:.2f} m/s")
    m3.metric("Direction", direction)
    m4.metric("Altitude", f"{row['altitude']:.1f} m")
    m5.metric("Threat", threat_level)
    m6.metric("Tracking", st.session_state.tracking_state)

    st.markdown("")

    # ========================================================
    # MAPS ROW
    # ========================================================

    map_left, map_right = st.columns([1.7, 1.0])

    # --------------------------------------------------------
    # DETAILED SATELLITE TRACKING MAP
    # --------------------------------------------------------
    with map_left:
        st.subheader("Satellite Tracking Map")

        tracking_map = folium.Map(
            location=[latitude, longitude],
            zoom_start=13,
            tiles=None,
            control_scale=True,
            zoom_control=True,
        )

        folium.TileLayer(
            tiles=(
                "https://services.arcgisonline.com/"
                "ArcGIS/rest/services/"
                "World_Imagery/MapServer/"
                "tile/{z}/{y}/{x}"
            ),
            attr=(
                "Esri, Maxar, Earthstar Geographics, "
                "and the GIS User Community"
            ),
            name="Satellite",
            overlay=False,
            control=False,
        ).add_to(tracking_map)

        tracking_layer = folium.FeatureGroup(name="UAV Tracking")

        path_data = data.iloc[: current_index + 1]

        flight_path = list(
            zip(
                path_data["latitude"].astype(float),
                path_data["longitude"].astype(float),
            )
        )

        if len(flight_path) > 1:
            folium.PolyLine(
                flight_path,
                weight=4,
                tooltip="UAV Flight Path",
            ).add_to(tracking_layer)

        folium.Circle(
            location=[zone_lat, zone_lon],
            radius=12,
            tooltip="Demo Protected Zone",
            popup="Demo Protected Zone",
            fill=True,
            fill_opacity=0.25,
        ).add_to(tracking_layer)

        folium.Marker(
            location=[latitude, longitude],
            tooltip="Current UAV Position",
            popup=(
                f"<b>TAIF-01</b>"
                f"<br>Speed: {row['speed']:.2f} m/s"
                f"<br>Altitude: {row['altitude']:.1f} m"
                f"<br>Threat: {threat_level}"
            ),
            icon=folium.Icon(icon="plane", prefix="fa"),
        ).add_to(tracking_layer)

        if st.session_state.auto_follow:
            st_folium(
                tracking_map,
                center=[latitude, longitude],
                zoom=13,
                feature_group_to_add=tracking_layer,
                height=480,
                use_container_width=True,
                key="detailed_tracking_map",
                returned_objects=[],
            )
        else:
            st_folium(
                tracking_map,
                feature_group_to_add=tracking_layer,
                height=480,
                use_container_width=True,
                key="detailed_tracking_map",
                returned_objects=[],
            )

    # --------------------------------------------------------
    # SAUDI OVERVIEW
    # --------------------------------------------------------
    with map_right:
        st.subheader("Saudi Arabia Overview")

        overview_map = folium.Map(
            location=[24.0, 45.0],
            zoom_start=5,
            tiles=None,
            control_scale=False,
            zoom_control=True,
        )

        folium.TileLayer(
            tiles=(
                "https://services.arcgisonline.com/"
                "ArcGIS/rest/services/"
                "World_Imagery/MapServer/"
                "tile/{z}/{y}/{x}"
            ),
            attr=(
                "Esri, Maxar, Earthstar Geographics, "
                "and the GIS User Community"
            ),
            name="Satellite",
            overlay=False,
            control=False,
        ).add_to(overview_map)

        if saudi_boundary is not None:
            folium.GeoJson(
                saudi_boundary,
                name="Saudi Arabia Boundary",
                style_function=lambda feature: {
                    "color": "#00d8ff",
                    "weight": 3,
                    "fillOpacity": 0,
                },
            ).add_to(overview_map)

        overview_layer = folium.FeatureGroup(name="Current UAV")

        folium.CircleMarker(
            location=[latitude, longitude],
            radius=7,
            tooltip="TAIF-01",
            fill=True,
            fill_opacity=1,
        ).add_to(overview_layer)

        st_folium(
            overview_map,
            center=[24.0, 45.0],
            zoom=5,
            feature_group_to_add=overview_layer,
            height=480,
            use_container_width=True,
            key="saudi_overview_map",
            returned_objects=[],
        )

    st.divider()

    # ========================================================
    # INFORMATION / STATUS ROW
    # ========================================================

    telemetry_col, status_col = st.columns([1.05, 1.0])

    # --------------------------------------------------------
    # TELEMETRY
    # --------------------------------------------------------
    with telemetry_col:
        st.subheader("UAV Telemetry")

        t1, t2 = st.columns(2)

        with t1:
            st.write(f"**Sample:** {current_index + 1} / {len(data)}")
            st.write(f"**Latitude:** {latitude:.7f}")
            st.write(f"**Longitude:** {longitude:.7f}")
            st.write(f"**Speed:** {row['speed']:.2f} m/s")
            st.write(f"**Heading:** {heading:.1f}°")

        with t2:
            st.write(f"**Altitude:** {row['altitude']:.1f} m")
            st.write(f"**LiDAR:** {row['lidar_range']:.1f} m")
            st.write(f"**AMSL:** {row['AMSL']:.1f} m")
            st.write(f"**Distance to Zone:** {distance_to_zone:.1f} m")
            st.write(f"**Timestamp:** {row['timestamp']:.2f}")

        st.subheader("Threat Assessment")

        if threat_level == "HIGH":
            st.error("HIGH THREAT")
            st.error(f"Recommended Action: {recommendation}")

        elif threat_level == "MEDIUM":
            st.warning("MEDIUM THREAT")
            st.warning(f"Recommended Action: {recommendation}")

        else:
            st.success("LOW THREAT")
            st.success(f"Recommended Action: {recommendation}")

    # --------------------------------------------------------
    # TRACKING + ALERTS
    # --------------------------------------------------------
    with status_col:
        st.subheader("Tracking Status")

        tracking_state = st.session_state.tracking_state

        if tracking_state == "TRACKING":
            st.success("🟢 TRACKING")

        elif tracking_state == "RE-ACQUIRING":
            st.warning("🟡 RE-ACQUIRING TARGET")

        elif tracking_state == "FAILED":
            st.error("🔴 RE-ACQUISITION FAILED")

        else:
            st.error(f"🔴 {tracking_state}")

        st.subheader("Alert Center")

        if tracking_state == "TRACKING":
            st.success("Target tracking active.")

        elif tracking_state == "RE-ACQUIRING":
            st.warning(
                "Target lost. Automatic re-acquisition in progress."
            )

        elif tracking_state == "FAILED":
            st.error(
                "Target could not be re-acquired. "
                "Operator decision required."
            )

        if threat_level == "HIGH":
            st.error(
                "Protected-zone alert: UAV is inside the HIGH threat threshold."
            )

        elif threat_level == "MEDIUM":
            st.warning(
                "Protected-zone alert: UAV is approaching the protected zone."
            )

    st.divider()

    # ========================================================
    # DECISION & SAFETY CONTROL
    # ========================================================

    st.subheader("Decision & Safety Control")

    decision_left, decision_right = st.columns([1.4, 1.0])

    with decision_left:

        if tracking_state == "TRACKING":
            st.success("System Status: Normal Tracking")
            st.info("Recommended Action: Continue Monitoring")

        elif tracking_state == "RE-ACQUIRING":
            st.warning(
                "Automatic Target Re-acquisition is in progress."
            )

        elif tracking_state == "FAILED":
            st.error("Automatic Re-acquisition Failed")
            st.warning("Recommended Action: Enter Safe State")

    with decision_right:

        if tracking_state == "FAILED":

            safe_col1, safe_col2 = st.columns(2)

            with safe_col1:
                if st.button(
                    "✅ ACCEPT SAFE STATE",
                    use_container_width=True,
                    key="accept_safe_state",
                ):
                    st.session_state.safe_state = True
                    st.session_state.safe_rejected = False

            with safe_col2:
                if st.button(
                    "❌ REJECT SAFE STATE",
                    use_container_width=True,
                    key="reject_safe_state",
                ):
                    st.session_state.safe_state = False
                    st.session_state.safe_rejected = True

    if st.session_state.safe_state:
        st.error("🔴 SAFE STATE ACTIVE")

        sfa, sfb, sfc = st.columns(3)
        sfa.metric("Autonomous Actions", "DISABLED")
        sfb.metric("Monitoring", "ACTIVE")
        sfc.metric("Operator Alert", "ACTIVE")

    elif st.session_state.safe_rejected:
        st.warning("Safe State recommendation rejected by operator.")

    st.divider()

    # ========================================================
    # SYSTEM STATUS BAR
    # ========================================================

    st.subheader("System Status")

    s1, s2, s3, s4, s5 = st.columns(5)

    s1.success("Internet Data\nCONNECTED")
    s2.success("Satellite Maps\nONLINE")
    s3.info("YOLO11s\nAWAITING")
    s4.info("BoT-SORT\nAWAITING")

    if st.session_state.safe_state:
        s5.error("Safety\nSAFE STATE")
    else:
        s5.success("Dashboard\nONLINE")

    st.caption(
        "FINAL PROTOTYPE MODE: Internet telemetry is real public dataset input. "
        "Tracking loss / re-acquisition are temporary demo logic. "
        "When the team outputs arrive, replace only the data/tracking source with "
        "YOLO11s + BoT-SORT outputs; the dashboard layout and safety workflow remain."
    )


dashboard_body()
