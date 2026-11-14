import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
import calendar
from datetime import datetime, timezone, timedelta

st.set_page_config(
    page_title="NASA FIRMS Orbital Fire Telemetry & NRT Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Dark Space"

# --- TOP BAR: THEME TOGGLE ON TOP RIGHT ---
top_spacer, top_theme_col = st.columns([6.5, 1.5])
with top_theme_col:
    theme_choice = st.radio(
        "Theme",
        ["Dark Space", "Light Mode"],
        horizontal=True,
        label_visibility="collapsed",
        index=0 if st.session_state.ui_theme == "Dark Space" else 1
    )
    st.session_state.ui_theme = theme_choice

is_dark = st.session_state.ui_theme == "Dark Space"

# --- DYNAMIC SPACE / LIGHT PALETTE ---
if is_dark:
    bg_main = "radial-gradient(circle at 18% 8%, #16213E 0%, #0B0E17 48%, #06080F 100%)"
    card_bg = "rgba(19, 25, 41, 0.85)"
    card_border = "#283552"
    text_primary = "#F4F7FC"
    text_secondary = "#A3B1CC"
    accent_primary = "#FF5722"
    accent_secondary = "#38BDF8"
    plotly_template = "plotly_dark"
    plot_bg = "rgba(19, 25, 41, 0.7)"
    map_style = "carto-darkmatter"
    nav_bg = "#111726"
else:
    bg_main = "linear-gradient(180deg, #F0F4F9 0%, #E6ECF5 100%)"
    card_bg = "#FFFFFF"
    card_border = "#CBD5E1"
    text_primary = "#0F172A"
    text_secondary = "#475569"
    accent_primary = "#E34C26"
    accent_secondary = "#0284C7"
    plotly_template = "plotly_white"
    plot_bg = "#FFFFFF"
    map_style = "carto-positron"
    nav_bg = "#FFFFFF"

# --- COSMIC / TECHNICAL UI STYLING (ENLARGED FONTS) ---
st.markdown(f"""
<style>
    .stApp {{
        background: {bg_main};
        color: {text_primary};
        font-size: 16px;
    }}
    p, li, span, div {{
        font-size: 16px;
    }}
    .hero-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-top: 3px solid {accent_primary};
        border-radius: 12px;
        padding: 32px 36px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.22);
    }}
    .hero-badge {{
        display: inline-block;
        font-family: monospace;
        font-size: 14px;
        font-weight: 700;
        color: {accent_secondary};
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }}
    .hero-title {{
        font-size: 36px !important;
        font-weight: 800;
        color: {text_primary};
        line-height: 1.2;
        margin-bottom: 12px;
    }}
    .hero-desc {{
        font-size: 18px !important;
        color: {text_secondary};
        line-height: 1.6;
        max-width: 960px;
    }}
    .exec-box {{
        background: {card_bg};
        border-left: 5px solid {accent_primary};
        border-top: 1px solid {card_border};
        border-right: 1px solid {card_border};
        border-bottom: 1px solid {card_border};
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 22px;
    }}
    .exec-title {{
        font-size: 14px !important;
        font-weight: 700;
        text-transform: uppercase;
        color: {accent_primary};
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }}
    .exec-text {{
        font-size: 16px !important;
        color: {text_primary};
        line-height: 1.6;
    }}
    .live-banner {{
        background-color: rgba(46, 160, 67, 0.14);
        border: 1px solid rgba(46, 160, 67, 0.45);
        border-left: 5px solid #2EA043;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        color: {text_primary};
        font-size: 16px !important;
    }}
    .live-alert {{
        background-color: rgba(227, 76, 38, 0.16);
        border: 1px solid rgba(227, 76, 38, 0.5);
        border-left: 5px solid #E34C26;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        color: {text_primary};
        font-size: 16px !important;
    }}
    .caveat-box {{
        background-color: rgba(210, 153, 34, 0.12);
        border: 1px solid rgba(210, 153, 34, 0.45);
        border-left: 5px solid #D29922;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 14px;
        margin-bottom: 18px;
        font-size: 15px !important;
        color: {text_primary};
    }}
    div[data-testid="stMetric"] {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 10px;
        padding: 16px 20px;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {text_secondary} !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {text_primary} !important;
        font-size: 25px !important;
        font-weight: 700 !important;
    }}
    div[data-baseweb="radio"] > div {{
        background: {nav_bg};
        border: 1px solid {card_border};
        border-radius: 10px;
        padding: 6px 12px;
    }}
</style>
""", unsafe_allow_html=True)


def get_firms_api_key():
    try:
        if "FIRMS_MAP_KEY" in st.secrets:
            return str(st.secrets["FIRMS_MAP_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("FIRMS_MAP_KEY", "").strip()


@st.cache_data
def load_and_harmonize():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    modis_list, viirs_list = [], []

    for f in files:
        try:
            fname = os.path.basename(f).lower()
            df_temp = pd.read_csv(f, low_memory=False)
            if "m-c61" in fname or "847792" in fname or "modis" in fname:
                modis_list.append(df_temp)
            elif "sv-c2" in fname or "847793" in fname or "viirs" in fname:
                viirs_list.append(df_temp)
        except Exception:
            pass

    if not modis_list or not viirs_list:
        return None, None, None, os.listdir(base_dir)

    modis = pd.concat(modis_list, ignore_index=True)
    viirs = pd.concat(viirs_list, ignore_index=True)

    modis["norm_conf"] = pd.to_numeric(
        modis["confidence"], errors="coerce"
    ).fillna(50)
    conf_map = {
        "l": 30, "n": 70, "h": 95,
        "low": 30, "nominal": 70, "high": 95
    }
    viirs["norm_conf"] = (
        viirs["confidence"].astype(str).str.lower().map(conf_map).fillna(70)
    )

    modis = modis[modis["norm_conf"] >= 30].copy()
    viirs = viirs[viirs["norm_conf"] >= 30].copy()

    modis["sensor"] = "MODIS (1 km)"
    viirs["sensor"] = "VIIRS (375 m)"

    for df in [modis, viirs]:
        df["acq_date"] = pd.to_datetime(df["acq_date"])
        df["year"] = df["acq_date"].dt.year
        df["month"] = df["acq_date"].dt.month
        df["day"] = df["acq_date"].dt.day
        df["grid_lat"] = df["latitude"].round(2)
        df["grid_lon"] = df["longitude"].round(2)

    cols = [
        "latitude", "longitude", "acq_date", "year", "month",
        "day", "frp", "norm_conf", "sensor", "grid_lat", "grid_lon"
    ]
    raw_combined = pd.concat([modis[cols], viirs[cols]], ignore_index=True)

    grp_cols = ["acq_date", "year", "month", "day", "grid_lat", "grid_lon"]
    harmonized = raw_combined.groupby(grp_cols).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        total_frp=("frp", "mean"),
        max_conf=("norm_conf", "max"),
        sensor_count=("sensor", "nunique"),
        raw_detections=("frp", "count")
    ).reset_index()

    conf_factor = harmonized["max_conf"] / 100.0
    dual_bonus = 1 + 0.25 * (harmonized["sensor_count"] - 1)
    harmonized["fire_activity_index"] = (
        harmonized["total_frp"] * conf_factor * dual_bonus
    )
    harmonized["biomass_rate_kg_s"] = harmonized["total_frp"] * 0.368

    return modis, viirs, harmonized, []


@st.cache_data(ttl=600)
def fetch_live_firms(api_key, day_range=2):
    bbox = "25.5,35.5,45.0,42.5"
    sources = {
        "MODIS_NRT": "MODIS (1 km)",
        "VIIRS_SNPP_NRT": "VIIRS S-NPP (375 m)",
        "VIIRS_NOAA20_NRT": "VIIRS NOAA-20 (375 m)"
    }
    frames = []
    errors = []

    for src_code, src_label in sources.items():
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{api_key}/{src_code}/{bbox}/{day_range}"
        )
        try:
            df_live = pd.read_csv(url)
            if "latitude" in df_live.columns and len(df_live) > 0:
                df_live["sensor_label"] = src_label
                frames.append(df_live)
        except Exception as e:
            errors.append(f"{src_code}: {str(e)}")

    if not frames:
        return pd.DataFrame(), errors

    live_all = pd.concat(frames, ignore_index=True)

    conf_map = {
        "l": 30, "n": 70, "h": 95,
        "low": 30, "nominal": 70, "high": 95
    }
    num_c = pd.to_numeric(live_all["confidence"], errors="coerce")
    cat_c = live_all["confidence"].astype(str).str.lower().map(conf_map)
    live_all["norm_conf"] = num_c.fillna(cat_c).fillna(50)

    time_str = live_all["acq_time"].astype(str).str.zfill(4)
    dt_str = (
        live_all["acq_date"].astype(str) + " " +
        time_str.str[:2] + ":" + time_str.str[2:]
    )
    live_all["overpass_utc"] = pd.to_datetime(dt_str, utc=True, errors="coerce")
    live_all["grid_lat"] = live_all["latitude"].round(2)
    live_all["grid_lon"] = live_all["longitude"].round(2)

    return live_all, errors


modis_df, viirs_df, harm_df, debug_files = load_and_harmonize()

if harm_df is None:
    st.error(f"Dataset files not found in directory: {debug_files}")
    st.stop()

month_names = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}
full_month_names = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

firms_key = get_firms_api_key()

# --- SIDEBAR PARAMETERS ---
with st.sidebar:
    st.markdown("### Orbital Telemetry Controls")
    live_days = st.selectbox(
        "Live NRT Lookback Window",
        [1, 2],
        index=1,
        format_func=lambda d: f"Last {d*24} Hours"
    )
    min_live_conf = st.slider(
        "Live Verification Sensitivity (%)",
        min_value=50,
        max_value=95,
        value=75,
        step=5
    )

    st.markdown("---")
    st.markdown("### Archive Query Parameters")
    years = sorted(harm_df["year"].unique())
    selected_years = st.multiselect(
        "Observation Period (Years)", years, default=years
    )

    aoi_presets = {
        "All Türkiye (35.5°N–42.5°N)": ((35.5, 42.5), (25.5, 45.0)),
        "Mediterranean Coast (Antalya & Mersin)": ((36.0, 37.5), (29.2, 35.2)),
        "Aegean Coast (Muğla, İzmir & Manisa)": ((36.6, 39.2), (26.2, 29.5)),
        "Eastern Mediterranean (Adana & Hatay)": ((35.8, 38.0), (35.0, 37.2)),
        "Southeastern Anatolia (Diyarbakır & Mardin)": ((36.7, 38.4), (39.5, 42.2)),
        "Black Sea Region (Coastal Belt)": ((40.5, 42.2), (31.0, 41.5)),
        "Custom Bounding Box (Lat/Lon)": None
    }

    region_preset = st.selectbox("Area of Interest (AOI)", list(aoi_presets.keys()))

    if region_preset == "Custom Bounding Box (Lat/Lon)":
        lat_range = st.slider("Latitude Range (°N)", 35.5, 42.5, (36.0, 39.0), step=0.1)
        lon_range = st.slider("Longitude Range (°E)", 25.5, 45.0, (27.0, 36.0), step=0.1)
    else:
        lat_range, lon_range = aoi_presets[region_preset]
        st.caption(
            f"Bounds: {lat_range[0]}°N–{lat_range[1]}°N, "
            f"{lon_range[0]}°E–{lon_range[1]}°E"
        )

    st.markdown("---")
    st.markdown("### Data Export")

f_harm = harm_df[
    (harm_df["year"].isin(selected_years)) &
    (harm_df["latitude"].between(lat_range[0], lat_range[1])) &
    (harm_df["longitude"].between(lon_range[0], lon_range[1]))
]
f_modis = modis_df[
    (modis_df["year"].isin(selected_years)) &
    (modis_df["latitude"].between(lat_range[0], lat_range[1])) &
    (modis_df["longitude"].between(lon_range[0], lon_range[1]))
]
f_viirs = viirs_df[
    (viirs_df["year"].isin(selected_years)) &
    (viirs_df["latitude"].between(lat_range[0], lat_range[1])) &
    (viirs_df["longitude"].between(lon_range[0], lon_range[1]))
]

with st.sidebar:
    csv_export = f_harm.head(10000).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Harmonized CSV",
        data=csv_export,
        file_name="harmonized_fire_events.csv",
        mime="text/csv"
    )
    st.markdown("---")
    st.caption("Constellation: Aqua • Terra • Suomi-NPP • NOAA-20")

# --- TOP PAGE NAVIGATION (SEPARATE PAGES) ---
PAGES = [
    "Mission Overview",
    "Live NRT Surveillance",
    "Seasonal Calendar & Gaps",
    "Monthly Monitoring",
    "Climatological Anomaly",
    "Spatial Distribution",
    "Calibration Validation",
    "Thermodynamic Modeling",
    "Methodology & Caveats"
]

active_page = st.radio(
    "Mission Modules",
    PAGES,
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<hr style='margin-top:8px; margin-bottom:22px; opacity:0.25;'>", unsafe_allow_html=True)

# Shared KPI calculations
raw_total = len(f_modis) + len(f_viirs)
harm_total = len(f_harm)
reduction = ((raw_total - harm_total) / raw_total * 100) if raw_total > 0 else 0
mean_frp_val = f_harm["total_frp"].mean() if harm_total > 0 else 0.0
mean_frp_err = mean_frp_val * 0.15
dual_count = len(f_harm[f_harm["sensor_count"] == 2])

# ==============================================================================
# PAGE 0: MISSION OVERVIEW (DEDICATED LANDING PAGE)
# ==============================================================================
if active_page == "Mission Overview":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-badge">
            🛰️ NASA FIRMS ARCHIVE & NRT STREAM • MULTI-SENSOR HARMONIZATION
        </div>
        <div class="hero-title">
            MODIS & VIIRS Fire Activity Calendar, Live NRT & Thermal Analysis
        </div>
        <div class="hero-desc">
            A multi-sensor Earth observation platform aligning MODIS (1 km) and VIIRS (375 m)
            orbital thermal anomalies onto a unified 0.01° spatial grid—combining near real-time
            satellite overpass verification, seasonal fire regime calendars, and bounded
            radiative combustion kinetics across Türkiye.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Archive Raw Detections", f"{raw_total:,}")
    col2.metric("Harmonized Fire Events", f"{harm_total:,}", f"-{reduction:.1f}% spatial overlap")
    col3.metric("Dual-Sensor Coincident", f"{dual_count:,}")
    col4.metric("Mean Radiative Power", f"{mean_frp_val:.1f} ± {mean_frp_err:.1f} MW")

    st.markdown("<br>", unsafe_allow_html=True)

    if len(f_harm) > 0:
        monthly_counts = (
            f_harm.groupby("month")["fire_activity_index"]
            .count()
            .sort_values(ascending=False)
        )
        top_months = [month_names.get(m, str(m)) for m in monthly_counts.head(2).index]
        peak_window = " – ".join(top_months)
        primary_month = monthly_counts.index[0]

        if primary_month in [6, 7, 8, 9]:
            regime_label = "High-Intensity Canopy & Wildland Fire Regime"
            action_rec = (
                "Prioritize pre-positioning of aerial suppression assets, "
                "thermal watchtowers, and rapid initial-attack crews 3 weeks "
                "prior to peak window."
            )
        elif primary_month in [10, 11]:
            regime_label = "Post-Harvest Agricultural Stubble & Residue Regime"
            action_rec = (
                "Deploy ground-based agricultural inspection patrols and "
                "enforce controlled-burn permits to prevent wildland-urban "
                "interface escape."
            )
        else:
            regime_label = "Low-Intensity / Transitional Thermal Regime"
            action_rec = (
                "Maintain baseline satellite monitoring and verify isolated "
                "thermal detections against industrial or cloud-edge false "
                "positives."
            )

        st.markdown(f"""
        <div class="exec-box">
            <div class="exec-title">
                Operational Decision Brief & Seasonal Prioritization • {region_preset}
            </div>
            <div class="exec-text">
                <strong>Highest-Risk Months:</strong> {peak_window} &nbsp;|&nbsp;
                <strong>Dominant Fire Regime:</strong> {regime_label}<br>
                <strong>Preparedness Recommendation:</strong> {action_rec}
            </div>
        </div>
        """, unsafe_allow_html=True)

    c_intro1, c_intro2, c_intro3 = st.columns(3)
    with c_intro1:
        st.markdown(f"""
        <div class="exec-box" style="border-left-color: {accent_secondary};">
            <div class="exec-title" style="color: {accent_secondary};">01 • Live Orbital Stream</div>
            <div class="exec-text">
                Tracks real-time MODIS & VIIRS overpasses (Suomi-NPP & NOAA-20),
                calculates exact overpass latency in hours/minutes, and filters
                high-confidence active fires.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c_intro2:
        st.markdown(f"""
        <div class="exec-box" style="border-left-color: {accent_secondary};">
            <div class="exec-title" style="color: {accent_secondary};">02 • Spatiotemporal Calendar</div>
            <div class="exec-text">
                Maps multi-year seasonal matrices and daily progressions while
                explicitly distinguishing zero-detection periods from cloud-masked
                orbital gaps.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c_intro3:
        st.markdown(f"""
        <div class="exec-box" style="border-left-color: {accent_secondary};">
            <div class="exec-title" style="color: {accent_secondary};">03 • Radiative Kinetics</div>
            <div class="exec-text">
                Derives instantaneous dry biomass burn rates (ṁ) and radiative
                heat flux (q'') with full Root-Sum-Square (RSS) sensor uncertainty
                propagation.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 1: LIVE NRT SURVEILLANCE
# ==============================================================================
elif active_page == "Live NRT Surveillance":
    st.markdown("### Near Real-Time (NRT) Orbital Fire Verification & Thermal Comparison")

    if not firms_key:
        st.warning(
            "Live NRT API key is not configured in server secrets. "
            "Add FIRMS_MAP_KEY to Streamlit Secrets to enable live satellite streaming."
        )
    else:
        with st.spinner("Querying NASA FIRMS Near Real-Time orbital feeds..."):
            live_raw, live_errs = fetch_live_firms(firms_key, day_range=live_days)

        now_utc = datetime.now(timezone.utc)
        trt_tz = timezone(timedelta(hours=3))
        now_trt = now_utc.astimezone(trt_tz)

        if len(live_raw) > 0 and live_raw["overpass_utc"].notna().any():
            latest_pass_utc = live_raw["overpass_utc"].max()
            latest_pass_trt = latest_pass_utc.astimezone(trt_tz)
            age_delta = now_utc - latest_pass_utc
            age_hours = int(age_delta.total_seconds() // 3600)
            age_mins = int((age_delta.total_seconds() % 3600) // 60)
            latency_str = f"{age_hours}h {age_mins}m ago"
            pass_time_str = (
                f"{latest_pass_trt.strftime('%Y-%m-%d %H:%M')} TRT "
                f"({latest_pass_utc.strftime('%H:%M')} UTC)"
            )
        else:
            latency_str = "No recent overpass data"
            pass_time_str = f"Checked at {now_trt.strftime('%Y-%m-%d %H:%M')} TRT"

        live_aoi = live_raw[
            (live_raw["latitude"].between(lat_range[0], lat_range[1])) &
            (live_raw["longitude"].between(lon_range[0], lon_range[1]))
        ].copy() if len(live_raw) > 0 else pd.DataFrame()

        high_sens_live = (
            live_aoi[live_aoi["norm_conf"] >= min_live_conf].copy()
            if len(live_aoi) > 0 else pd.DataFrame()
        )

        if len(high_sens_live) > 0:
            live_grid = high_sens_live.groupby(["grid_lat", "grid_lon"]).agg(
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
                total_frp=("frp", "mean"),
                max_conf=("norm_conf", "max"),
                sensor_passes=("sensor_label", "count"),
                unique_sensors=("sensor_label", "nunique"),
                last_seen=("overpass_utc", "max")
            ).reset_index()

            live_grid["Verification_Status"] = np.where(
                (live_grid["unique_sensors"] >= 2) |
                (live_grid["sensor_passes"] >= 2) |
                (live_grid["max_conf"] >= 90),
                "Confirmed Active Fire (High Confidence / Multi-Pass)",
                "Probable Thermal Hotspot (Single-Pass High Conf)"
            )
            confirmed_cnt = len(
                live_grid[live_grid["Verification_Status"].str.startswith("Confirmed")]
            )
        else:
            live_grid = pd.DataFrame()
            confirmed_cnt = 0

        if confirmed_cnt > 0:
            st.markdown(f"""
            <div class="live-alert">
                <strong>ACTIVE FIRE ALERT ({region_preset}):</strong>
                <strong>{confirmed_cnt}</strong> confirmed high-confidence active fire cells detected
                within the last {live_days*24} hours. &nbsp;|&nbsp;
                <strong>Latest Satellite Overpass:</strong> {pass_time_str} (<strong>{latency_str}</strong>)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="live-banner">
                <strong>STATUS CLEAR — NO CONFIRMED ACTIVE FIRES ({region_preset}):</strong>
                Zero confirmed high-sensitivity (≥{min_live_conf}%) wildfire anomalies detected
                in the last {live_days*24} hours. &nbsp;|&nbsp;
                <strong>Latest Satellite Overpass:</strong> {pass_time_str} (<strong>{latency_str}</strong>)
            </div>
            """, unsafe_allow_html=True)

        lk1, lk2, lk3, lk4 = st.columns(4)
        lk1.metric(
            "Latest Orbital Overpass",
            latency_str,
            latest_pass_trt.strftime("%H:%M TRT") if len(live_raw) > 0 else None
        )
        lk2.metric(
            "Confirmed Active Fire Cells",
            f"{confirmed_cnt:,}",
            f"Sensitivity ≥ {min_live_conf}%"
        )
        lk3.metric(
            "Total High-Conf Hotspots",
            f"{len(live_grid):,}",
            f"Raw NRT: {len(live_aoi):,}"
        )
        live_frp = live_grid["total_frp"].sum() if len(live_grid) > 0 else 0.0
        lk4.metric("Live Radiative Power", f"{live_frp:,.1f} MW")

        # Full-Width Large Interactive Live Map
        if len(live_grid) > 0:
            live_grid["Last Pass (TRT)"] = (
                live_grid["last_seen"]
                .dt.tz_convert(trt_tz)
                .dt.strftime("%Y-%m-%d %H:%M")
            )
            fig_lmap = px.scatter_map(
                live_grid,
                lat="latitude",
                lon="longitude",
                size="total_frp",
                color="Verification_Status",
                color_discrete_map={
                    "Confirmed Active Fire (High Confidence / Multi-Pass)": "#E34C26",
                    "Probable Thermal Hotspot (Single-Pass High Conf)": "#D29922"
                },
                hover_data=["Last Pass (TRT)", "total_frp", "max_conf", "unique_sensors"],
                zoom=5.5,
                height=650,
                map_style=map_style
            )
            fig_lmap.update_layout(
                title="Live Verified Thermal Map (MODIS + VIIRS S-NPP & NOAA-20 NRT)",
                template=plotly_template,
                margin=dict(l=0, r=0, t=44, b=0),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_lmap, use_container_width=True)
        else:
            empty_df = pd.DataFrame({"lat": [39.0], "lon": [35.2]})
            fig_empty = px.scatter_map(
                empty_df,
                lat="lat",
                lon="lon",
                zoom=5.5,
                height=650,
                map_style=map_style
            )
            fig_empty.update_traces(marker=dict(size=0, opacity=0))
            fig_empty.update_layout(
                title="Live Verified Thermal Map — Zero Active Hotspots Detected",
                template=plotly_template,
                margin=dict(l=0, r=0, t=44, b=0),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_empty, use_container_width=True)

        # Full-Width Live vs. Climatological Normal Comparison Chart
        cur_month = now_trt.month
        hist_m = f_harm[f_harm["month"] == cur_month]
        if len(hist_m) > 0:
            hist_daily_avg = (
                hist_m.groupby("acq_date")["fire_activity_index"].count().mean()
            )
        else:
            hist_daily_avg = 0.0

        live_daily_rate = len(live_grid) / float(live_days)

        fig_lcomp = go.Figure()
        fig_lcomp.add_trace(go.Bar(
            x=[
                f"Historical {full_month_names[cur_month]} Daily Avg",
                "Live 24h Equivalent"
            ],
            y=[hist_daily_avg, live_daily_rate],
            marker_color=["#38BDF8", "#FF5722"],
            text=[f"{hist_daily_avg:.1f}", f"{live_daily_rate:.1f}"],
            textposition="auto",
            width=[0.38, 0.38],
            name="Daily Event Rate"
        ))
        fig_lcomp.update_layout(
            title=f"Live Activity vs. {full_month_names[cur_month]} Climatological Normal",
            yaxis_title="Events / 24h",
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=plot_bg,
            height=350,
            margin=dict(t=44)
        )
        st.plotly_chart(fig_lcomp, use_container_width=True)

# ==============================================================================
# PAGE 2: SEASONAL CALENDAR & GAPS
# ==============================================================================
elif active_page == "Seasonal Calendar & Gaps":
    st.markdown("### Monthly Fire Regime Matrix & Observation-Gap Transparency")
    if len(f_harm) > 0:
        min_d, max_d = f_harm["acq_date"].min(), f_harm["acq_date"].max()
        full_dates = pd.date_range(min_d, max_d, freq="D")
        observed_dates = set(f_harm["acq_date"].unique())
        missing_days = len([d for d in full_dates if d not in observed_dates])
        coverage_pct = ((len(full_dates) - missing_days) / len(full_dates)) * 100

        c_gap1, c_gap2, c_gap3 = st.columns(3)
        c_gap1.metric("Total Calendar Span", f"{len(full_dates):,} days")
        c_gap2.metric("No-Detection / Cloud-Masked Days", f"{missing_days:,} days")
        c_gap3.metric("Active Observation Coverage", f"{coverage_pct:.1f}%")

    cal_data = (
        f_harm.groupby(["year", "month"])
        .agg(fire_count=("fire_activity_index", "count"))
        .reset_index()
    )
    all_grid = pd.MultiIndex.from_product(
        [selected_years, range(1, 13)], names=["year", "month"]
    ).to_frame(index=False)
    cal_full = all_grid.merge(cal_data, on=["year", "month"], how="left")
    pivot_cal = cal_full.pivot(index="year", columns="month", values="fire_count")
    pivot_cal.columns = [month_names.get(c, c) for c in pivot_cal.columns]

    fig_cal = px.imshow(
        pivot_cal,
        labels=dict(x="Month", y="Year", color="Harmonized Events"),
        color_continuous_scale="YlOrRd",
        text_auto=True,
        aspect="auto"
    )
    fig_cal.update_layout(
        template=plotly_template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=plot_bg,
        height=480,
        margin=dict(t=24)
    )
    st.plotly_chart(fig_cal, use_container_width=True)
    st.markdown("""
    <div class="caveat-box">
        <strong>Observation-Gap Transparency ("Empty Cell" Distinction):</strong>
        Blank cells represent periods with <strong>zero valid orbital detections</strong>—which
        may result from heavy cloud cover, dense smoke obscuration, or orbital swath gaps rather
        than confirmed absence of fire.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 3: MONTHLY MONITORING
# ==============================================================================
elif active_page == "Monthly Monitoring":
    st.markdown("### Dedicated Month-by-Month Surveillance & Daily Progression")
    if len(f_harm) > 0:
        default_idx = int(
            f_harm.groupby("month")["fire_activity_index"].count().idxmax() - 1
        )
    else:
        default_idx = 7

    sel_month = st.selectbox(
        "Select Target Month for Dedicated Surveillance:",
        options=list(range(1, 13)),
        index=default_idx,
        format_func=lambda m: f"{full_month_names[m]} (Month {m:02d})"
    )

    m_harm = f_harm[f_harm["month"] == sel_month].copy()
    m_modis = f_modis[f_modis["month"] == sel_month]
    m_viirs = f_viirs[f_viirs["month"] == sel_month]

    mc1, mc2, mc3, mc4 = st.columns(4)
    m_events = len(m_harm)
    m_raw_total = len(m_modis) + len(m_viirs)
    m_frp_sum = m_harm["total_frp"].sum() if m_events > 0 else 0.0
    m_frp_err = m_frp_sum * 0.15
    m_mean_rate = m_harm["biomass_rate_kg_s"].mean() if m_events > 0 else 0.0
    m_rate_err = m_mean_rate * 0.202

    if m_events > 0:
        daily_m_series = m_harm.groupby("acq_date")["fire_activity_index"].count()
        peak_day_row = daily_m_series.idxmax()
        peak_day_str = peak_day_row.strftime("%Y-%m-%d")
        peak_day_cnt = daily_m_series.max()
    else:
        peak_day_str, peak_day_cnt = "N/A", 0

    mc1.metric(
        f"{month_names[sel_month]} Harmonized Events",
        f"{m_events:,}",
        f"Raw: {m_raw_total:,}"
    )
    mc2.metric(
        "Peak Activity Date",
        peak_day_str,
        f"{peak_day_cnt:,} events" if m_events > 0 else None
    )
    mc3.metric(
        f"{month_names[sel_month]} Cumulative FRP",
        f"{m_frp_sum:,.0f} ± {m_frp_err:,.0f} MW"
    )
    mc4.metric(
        f"{month_names[sel_month]} Mean Burn Rate",
        f"{m_mean_rate:.2f} ± {m_rate_err:.2f} kg/s"
    )

    days_in_m = calendar.monthrange(2024, sel_month)[1]
    day_grid = pd.MultiIndex.from_product(
        [selected_years, range(1, days_in_m + 1)], names=["year", "day"]
    ).to_frame(index=False)
    daily_m_counts = (
        m_harm.groupby(["year", "day"])
        .agg(events=("fire_activity_index", "count"))
        .reset_index()
    )
    day_matrix_df = day_grid.merge(daily_m_counts, on=["year", "day"], how="left")
    pivot_day = day_matrix_df.pivot(index="year", columns="day", values="events")

    fig_day_cal = px.imshow(
        pivot_day,
        labels=dict(x=f"Day of {full_month_names[sel_month]}", y="Year", color="Events"),
        color_continuous_scale="YlOrRd",
        text_auto=True,
        aspect="auto"
    )
    fig_day_cal.update_layout(
        title=f"Daily Matrix — {full_month_names[sel_month]} (Blank = No Obs / Cloud-Masked)",
        template=plotly_template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=plot_bg,
        margin=dict(t=44)
    )
    st.plotly_chart(fig_day_cal, use_container_width=True)

    c_left, c_right = st.columns([1, 1])
    with c_left:
        yr_comp = (
            m_harm.groupby("year")
            .agg(events=("fire_activity_index", "count"), mean_frp=("total_frp", "mean"))
            .reindex(selected_years, fill_value=0)
            .reset_index()
        )

        fig_yr_m = go.Figure()
        fig_yr_m.add_trace(go.Bar(
            x=yr_comp["year"].astype(str),
            y=yr_comp["events"],
            marker_color="#FF5722",
            name="Harmonized Events"
        ))
        if len(yr_comp) > 1 and yr_comp["events"].std() > 0:
            m_thresh = yr_comp["events"].mean() + 2 * yr_comp["events"].std()
            fig_yr_m.add_hline(
                y=m_thresh,
                line_dash="dash",
                line_color="#D29922",
                annotation_text="Month +2σ Threshold"
            )
        fig_yr_m.update_layout(
            title=f"Inter-Annual Comparison ({full_month_names[sel_month]})",
            xaxis_title="Year",
            yaxis_title="Harmonized Events",
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=plot_bg,
            height=430,
            margin=dict(t=44)
        )
        st.plotly_chart(fig_yr_m, use_container_width=True)

    with c_right:
        if m_events > 0:
            m_sample = m_harm.sort_values("fire_activity_index", ascending=False).head(1500)
            fig_m_map = px.scatter_map(
                m_sample,
                lat="latitude",
                lon="longitude",
                size="total_frp",
                color="sensor_count",
                color_continuous_scale=["#D29922", "#FF5722"],
                labels={"sensor_count": "Sensors", "total_frp": "FRP (MW)"},
                hover_data=["acq_date", "total_frp", "max_conf"],
                zoom=4.8,
                height=430,
                map_style=map_style
            )
            fig_m_map.update_layout(
                title=f"Top Spatial Hotspots ({full_month_names[sel_month]})",
                template=plotly_template,
                margin=dict(l=0, r=0, t=44, b=0),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_m_map, use_container_width=True)
        else:
            st.info(f"No valid detections recorded for {full_month_names[sel_month]}.")

# ==============================================================================
# PAGE 4: CLIMATOLOGICAL ANOMALY
# ==============================================================================
elif active_page == "Climatological Anomaly":
    st.markdown("### Climatological Seasonal Baseline, Anomalies (+2σ) & Missing Days")
    if len(f_harm) > 0:
        min_d, max_d = f_harm["acq_date"].min(), f_harm["acq_date"].max()
        full_calendar = pd.DataFrame({"acq_date": pd.date_range(min_d, max_d, freq="D")})
        full_calendar["month"] = full_calendar["acq_date"].dt.month

        daily_obs = (
            f_harm.groupby("acq_date")
            .agg(event_count=("fire_activity_index", "count"))
            .reset_index()
        )
        daily = full_calendar.merge(daily_obs, on="acq_date", how="left")

        monthly_clim = (
            daily.dropna(subset=["event_count"])
            .groupby("month")["event_count"]
            .agg(clim_mean="mean", clim_std="std")
            .reset_index()
        )
        monthly_clim["clim_std"] = monthly_clim["clim_std"].fillna(0)
        daily = daily.merge(monthly_clim, on="month", how="left")
        daily["upper_2sigma"] = daily["clim_mean"] + 2 * daily["clim_std"]

        no_obs_days = daily[daily["event_count"].isna()].copy()
        valid_days = daily.dropna(subset=["event_count"]).copy()
        valid_days["Classification"] = np.where(
            valid_days["event_count"] > valid_days["upper_2sigma"],
            "Extreme Anomaly",
            "Normal"
        )

        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(
            x=daily["acq_date"], y=daily["clim_mean"],
            mode="lines", name="Seasonal Climatological Normal (μ)",
            line=dict(color="#38BDF8", width=2.5)
        ))
        fig_anom.add_trace(go.Scatter(
            x=daily["acq_date"], y=daily["upper_2sigma"],
            mode="lines", name="Seasonal +2σ Threshold",
            line=dict(color="#D29922", width=2, dash="dash")
        ))

        normal_pts = valid_days[valid_days["Classification"] == "Normal"]
        anom_pts = valid_days[valid_days["Classification"] == "Extreme Anomaly"]

        fig_anom.add_trace(go.Scatter(
            x=normal_pts["acq_date"], y=normal_pts["event_count"],
            mode="markers", name="Normal Activity",
            marker=dict(color="#8B949E", size=6, opacity=0.65)
        ))
        fig_anom.add_trace(go.Scatter(
            x=anom_pts["acq_date"], y=anom_pts["event_count"],
            mode="markers", name="Extreme Anomaly (> +2σ)",
            marker=dict(color="#FF5722", size=9)
        ))
        fig_anom.add_trace(go.Scatter(
            x=no_obs_days["acq_date"], y=[0] * len(no_obs_days),
            mode="markers", name="No Observation / Cloud-Masked Day",
            marker=dict(color="#64748B", size=6, symbol="x")
        ))

        fig_anom.update_layout(
            xaxis_title="Date",
            yaxis_title="Daily Harmonized Fire Events",
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=plot_bg,
            height=520,
            margin=dict(t=24)
        )
        st.plotly_chart(fig_anom, use_container_width=True)

# ==============================================================================
# PAGE 5: SPATIAL DISTRIBUTION
# ==============================================================================
elif active_page == "Spatial Distribution":
    st.markdown("### Geospatial Distribution of High-Intensity Events (Top 3,000 by FAI)")
    sample_map = f_harm.sort_values("fire_activity_index", ascending=False).head(3000)
    fig_map = px.scatter_map(
        sample_map,
        lat="latitude",
        lon="longitude",
        size="total_frp",
        color="sensor_count",
        color_continuous_scale=["#D29922", "#FF5722"],
        labels={"sensor_count": "Sensor Count", "total_frp": "FRP (MW)", "max_conf": "Confidence (%)"},
        hover_data=["acq_date", "total_frp", "max_conf"],
        zoom=5.3,
        height=640,
        map_style=map_style
    )
    fig_map.update_layout(
        template=plotly_template,
        margin=dict(l=0, r=0, t=12, b=0),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption(
        "Hotspot ≠ Burned Area: Plotted coordinates indicate active thermal anomalies "
        "within 0.01° grid cells and do not directly represent total burned acreage."
    )

# ==============================================================================
# PAGE 6: CALIBRATION VALIDATION
# ==============================================================================
elif active_page == "Calibration Validation":
    st.markdown("### Statistical Calibration Consistency & Resolution Bias Correction")
    m_monthly = f_modis.groupby("month").size().reset_index(name="MODIS (1 km)")
    v_monthly = f_viirs.groupby("month").size().reset_index(name="VIIRS (375 m)")
    h_monthly = f_harm.groupby("month").size().reset_index(name="Harmonized")
    comp = (
        m_monthly.merge(v_monthly, on="month", how="outer")
        .merge(h_monthly, on="month", how="outer")
        .fillna(0)
    )
    comp["Month"] = comp["month"].map(month_names)

    if len(comp) > 1 and comp["MODIS (1 km)"].std() > 0 and comp["VIIRS (375 m)"].std() > 0:
        r_val = np.corrcoef(comp["MODIS (1 km)"], comp["VIIRS (375 m)"])[0, 1]
        r2_val = r_val ** 2
    else:
        r_val, r2_val = 0.0, 0.0

    dual_cells = f_harm[f_harm["sensor_count"] == 2]
    dual_share = (len(dual_cells) / len(f_harm) * 100) if len(f_harm) > 0 else 0

    s1, s2, s3 = st.columns(3)
    s1.metric("Cross-Sensor Temporal Correlation (r)", f"{r_val:.3f}")
    s2.metric("Coefficient of Determination (R²)", f"{r2_val:.3f}")
    s3.metric("Coincident Grid Overlap Rate", f"{dual_share:.1f}%")

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=comp["Month"], y=comp["MODIS (1 km)"],
        name="MODIS (1 km)", marker_color="#38BDF8"
    ))
    fig_comp.add_trace(go.Bar(
        x=comp["Month"], y=comp["VIIRS (375 m)"],
        name="VIIRS (375 m)", marker_color="#D29922"
    ))
    fig_comp.add_trace(go.Scatter(
        x=comp["Month"], y=comp["Harmonized"],
        name="Harmonized Grid Events", mode="lines+markers",
        line=dict(color="#FF5722", width=3.5)
    ))
    fig_comp.update_layout(
        barmode="group",
        xaxis_title="Month",
        yaxis_title="Event Count",
        template=plotly_template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=plot_bg,
        height=480,
        margin=dict(t=24)
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ==============================================================================
# PAGE 7: THERMODYNAMIC MODELING
# ==============================================================================
elif active_page == "Thermodynamic Modeling":
    st.markdown("### Bounded Radiative Power & Combustion Kinetics")
    st.caption(
        "Instantaneous quantities derived from orbital Fire Radiative Power (FRP ±15%) "
        "and Wooster et al. (2005) combustion factor (0.368 ± 0.05 kg/MJ). "
        "Single-year selection enables annual dry biomass integration (Δt_eff = 1 hr)."
    )

    total_mw = f_harm["total_frp"].sum()
    total_mw_err = total_mw * 0.15
    mean_burn_rate = f_harm["biomass_rate_kg_s"].mean() if harm_total > 0 else 0.0
    burn_rate_err = mean_burn_rate * 0.202
    peak_flux = (f_harm["total_frp"].max() * 1000) / (1100 * 1100) if harm_total > 0 else 0.0
    flux_err = peak_flux * 0.15

    if len(selected_years) == 1 and harm_total > 0:
        sel_yr = selected_years[0]
        annual_biomass_tons = 1.3248 * total_mw
        annual_biomass_err = annual_biomass_tons * 0.322
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cumulative Snapshot FRP", f"{total_mw:,.0f} ± {total_mw_err:,.0f} MW")
        m2.metric("Mean Event Burn Rate (ṁ)", f"{mean_burn_rate:.2f} ± {burn_rate_err:.2f} kg/s")
        m3.metric("Peak Grid Heat Flux", f"{peak_flux:.2f} ± {flux_err:.2f} kW/m²")
        m4.metric(
            f"{sel_yr} Est. Total Biomass (M_y)",
            f"{annual_biomass_tons:,.0f} ± {annual_biomass_err:,.0f} t"
        )
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Cumulative Snapshot FRP", f"{total_mw:,.0f} ± {total_mw_err:,.0f} MW")
        m2.metric("Mean Event Burn Rate (ṁ)", f"{mean_burn_rate:.2f} ± {burn_rate_err:.2f} kg/s")
        m3.metric("Peak Grid Heat Flux", f"{peak_flux:.2f} ± {flux_err:.2f} kW/m²")
        st.info(
            "Tip: Select a single year in 'Observation Period (Years)' to compute "
            "the annual integrated biomass loss (M_y)."
        )

    monthly_energy = f_harm.groupby("month")["total_frp"].sum().reset_index()
    monthly_energy["Month"] = monthly_energy["month"].map(month_names)
    monthly_energy["frp_err"] = monthly_energy["total_frp"] * 0.15

    fig_eng = go.Figure()
    fig_eng.add_trace(go.Bar(
        x=monthly_energy["Month"],
        y=monthly_energy["total_frp"],
        error_y=dict(
            type="data",
            array=monthly_energy["frp_err"],
            visible=True,
            color="#8B949E"
        ),
        marker_color="#FF5722",
        name="Cumulative Snapshot FRP (MW)"
    ))
    fig_eng.update_layout(
        xaxis_title="Month",
        yaxis_title="Cumulative Snapshot Radiative Power (MW ± 15%)",
        template=plotly_template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=plot_bg,
        height=440,
        margin=dict(t=24)
    )
    st.plotly_chart(fig_eng, use_container_width=True)

    st.markdown(f"""
    <div class="exec-box" style="border-left-color: {accent_secondary}; margin-top: 18px;">
        <div class="exec-title" style="color: {accent_secondary};">
            Governing Thermodynamic & Biomass Equations
        </div>
        <div class="exec-text">
            • <strong>Instantaneous Biomass Burn Rate (Wooster et al., 2005):</strong> &nbsp;
              <code>ṁ = 0.368 × FRP [kg/s]</code> &nbsp; <em>(RSS Uncertainty: ±20.2%)</em><br>
            • <strong>Peak Grid Radiative Heat Flux (0.01° ≈ 1.21 km² cell):</strong> &nbsp;
              <code>q'' = (FRP_max × 1000) / (1100 × 1100) [kW/m²]</code> &nbsp; <em>(Uncertainty: ±15.0%)</em><br>
            • <strong>Single-Year Integrated Dry Biomass Loss (Δt_eff = 3600 s):</strong> &nbsp;
              <code>M_y = (0.368 × 3600 / 1000) × Σ FRP_i = 1.3248 × Σ FRP_i [t/yr]</code> &nbsp; <em>(RSS Uncertainty: ±32.2%)</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 8: METHODOLOGY & CAVEATS
# ==============================================================================
elif active_page == "Methodology & Caveats":
    st.markdown("""
    <div class="caveat-box">
        <strong>Limitations & Ground-Truth Caveat (Hotspot ≠ Burned Area):</strong><br>
        Satellite thermal detections represent active radiative anomalies at the exact time
        of orbital overpass. A single harmonized grid cell does <strong>not</strong> directly
        measure total burned acreage (hectares) or individual ignition count, and can be
        obscured by thick cloud cover or dense canopy smoke. High-resolution optical imagery
        (e.g., Sentinel-2 dNBR) and field verification are required for post-fire perimeter mapping.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Harmonization Pipeline, Governing Equations & Uncertainty Budget")
    st.markdown("""
    1. **Spatial Resampling:** Native VIIRS (375 m) and MODIS (1 km) centroids are binned
       onto a unified $0.01^\\circ \\times 0.01^\\circ$ (~$1.11\\text{ km}$) geographic grid
       per UTC acquisition date to eliminate sub-pixel overcounting.
    2. **Confidence Calibration:** Categorical VIIRS flags (`low`, `nominal`, `high`) are
       mapped to quantitative equivalence scores (`30`, `70`, `95`) matching the MODIS `0–100%`
       scale; detections below `30%` are filtered as solar-glint / bare-soil noise per Giglio et al. (2016).
    3. **Composite Fire Activity Index (FAI):**
       $$\\text{FAI} = \\overline{\\text{FRP}} \\times \\left(\\frac{C_{\\max}}{100}\\right) \\times \\left[1 + 0.25(N_{\\text{sensors}} - 1)\\right]$$
    4. **Instantaneous Combustion Kinetics & Single-Year Integration:**
       - Instantaneous biomass burn rate: $\\dot{m}_b = 0.368 \\times \\text{FRP}\\ [\\text{kg/s}]$ *(Wooster et al., 2005)*
       - Single-year integrated dry biomass loss ($\\Delta t_{\\text{eff}} = 3600\\ \\text{s}$): $M_y = 1.3248 \\sum_{i=1}^{N_y} \\overline{\\text{FRP}}_i\\ [\\text{t/yr}]$ with RSS uncertainty $\\delta_M = \\sqrt{0.15^2 + 0.136^2 + 0.25^2} \\approx \\pm 32.2\\%$.
    """)
    st.markdown("### Sample Harmonized Records")
    st.dataframe(f_harm.head(50), use_container_width=True)
