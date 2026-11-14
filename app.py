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
top_spacer, top_theme_col = st.columns([6.2, 1.8])
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

# --- DYNAMIC HIGH-CONTRAST SPACE / LIGHT PALETTE ---
if is_dark:
    bg_main = "radial-gradient(circle at 18% 8%, #16213E 0%, #0B0E17 48%, #06080F 100%)"
    sidebar_bg = "#0D121F"
    card_bg = "rgba(19, 25, 41, 0.90)"
    card_border = "#2D3B59"
    text_primary = "#F8FAFC"
    text_secondary = "#CBD5E1"
    accent_primary = "#FF5722"
    accent_secondary = "#38BDF8"
    plotly_template = "plotly_dark"
    plot_bg = "rgba(19, 25, 41, 0.85)"
    chart_font_color = "#F8FAFC"
    map_style = "carto-darkmatter"
    nav_bg = "#131A2B"
    input_bg = "#161E31"
else:
    bg_main = "#F1F5F9"
    sidebar_bg = "#E2E8F0"
    card_bg = "#FFFFFF"
    card_border = "#94A3B8"
    text_primary = "#090D16"
    text_secondary = "#1E293B"
    accent_primary = "#D9381E"
    accent_secondary = "#0369A1"
    plotly_template = "plotly_white"
    plot_bg = "#FFFFFF"
    chart_font_color = "#090D16"
    map_style = "carto-positron"
    nav_bg = "#FFFFFF"
    input_bg = "#FFFFFF"

# --- GLOBAL ENLARGED TYPOGRAPHY & HIGH-CONTRAST THEME CSS ---
st.markdown(f"""
<style>
    .stApp {{
        background: {bg_main} !important;
        color: {text_primary} !important;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {card_border} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {text_primary} !important;
        font-size: 16px !important;
    }}
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, .stText, label {{
        color: {text_primary} !important;
        font-size: 18px !important;
        line-height: 1.65 !important;
    }}
    h1 {{
        color: {text_primary} !important;
        font-size: 38px !important;
        font-weight: 800 !important;
    }}
    h2 {{
        color: {text_primary} !important;
        font-size: 30px !important;
        font-weight: 700 !important;
    }}
    h3 {{
        color: {text_primary} !important;
        font-size: 25px !important;
        font-weight: 700 !important;
        margin-bottom: 14px !important;
    }}
    h4, h5 {{
        color: {text_primary} !important;
        font-size: 21px !important;
        font-weight: 700 !important;
    }}
    .stCaption, [data-testid="stCaptionContainer"] p {{
        color: {text_secondary} !important;
        font-size: 15px !important;
    }}
    /* Radio buttons & top navigation */
    div[data-baseweb="radio"] > div {{
        background: {nav_bg} !important;
        border: 1.5px solid {card_border} !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
    }}
    div[data-baseweb="radio"] label p {{
        color: {text_primary} !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }}
    /* Selectbox & Multiselect high contrast */
    div[data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        color: {text_primary} !important;
        border: 1.5px solid {card_border} !important;
        font-size: 16px !important;
    }}
    div[data-baseweb="select"] span, div[data-baseweb="select"] div {{
        color: {text_primary} !important;
        font-size: 16px !important;
    }}
    /* Hero & Cards */
    .hero-card {{
        background: {card_bg};
        border: 1.5px solid {card_border};
        border-top: 4px solid {accent_primary};
        border-radius: 12px;
        padding: 34px 38px;
        margin-bottom: 26px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.14);
    }}
    .hero-badge {{
        display: inline-block;
        font-family: monospace;
        font-size: 15px !important;
        font-weight: 700 !important;
        color: {accent_secondary} !important;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }}
    .hero-title {{
        font-size: 38px !important;
        font-weight: 800 !important;
        color: {text_primary} !important;
        line-height: 1.25 !important;
        margin-bottom: 14px;
    }}
    .hero-desc {{
        font-size: 19px !important;
        color: {text_secondary} !important;
        line-height: 1.65 !important;
        max-width: 1000px;
    }}
    .exec-box {{
        background: {card_bg};
        border-left: 5px solid {accent_primary};
        border-top: 1.5px solid {card_border};
        border-right: 1.5px solid {card_border};
        border-bottom: 1.5px solid {card_border};
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 22px;
    }}
    .exec-title {{
        font-size: 16px !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        color: {accent_primary} !important;
        letter-spacing: 0.8px;
        margin-bottom: 10px;
    }}
    .exec-text {{
        font-size: 18px !important;
        color: {text_primary} !important;
        line-height: 1.65 !important;
    }}
    .live-banner {{
        background-color: rgba(46, 160, 67, 0.16);
        border: 1.5px solid #2EA043;
        border-left: 6px solid #2EA043;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 22px;
        color: {text_primary} !important;
        font-size: 18px !important;
        line-height: 1.6 !important;
    }}
    .live-alert {{
        background-color: rgba(227, 76, 38, 0.16);
        border: 1.5px solid #E34C26;
        border-left: 6px solid #E34C26;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 22px;
        color: {text_primary} !important;
        font-size: 18px !important;
        line-height: 1.6 !important;
    }}
    .caveat-box {{
        background-color: rgba(210, 153, 34, 0.14);
        border: 1.5px solid #D29922;
        border-left: 6px solid #D29922;
        border-radius: 10px;
        padding: 18px 22px;
        margin-top: 16px;
        margin-bottom: 20px;
        font-size: 17px !important;
        color: {text_primary} !important;
        line-height: 1.6 !important;
    }}
    /* Metric Cards */
    div[data-testid="stMetric"] {{
        background: {card_bg} !important;
        border: 1.5px solid {card_border} !important;
        border-radius: 10px !important;
        padding: 18px 22px !important;
    }}
    div[data-testid="stMetricLabel"] p {{
        color: {text_secondary} !important;
        font-size: 16px !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stMetricValue"] div {{
        color: {text_primary} !important;
        font-size: 28px !important;
        font-weight: 800 !important;
    }}
    div[data-testid="stMetricDelta"] div {{
        font-size: 15px !important;
        font-weight: 600 !important;
    }}
    /* Download button high contrast */
    .stDownloadButton button {{
        background-color: {accent_primary} !important;
        color: #FFFFFF !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 10px 18px !important;
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
    9: "September",
