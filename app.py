import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
import calendar

st.set_page_config(
    page_title="MODIS-VIIRS Fire Regime & Thermal Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CLEAN TECHNICAL UI STYLING ---
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .header-container {
        border-bottom: 1px solid #262C36;
        padding-bottom: 16px;
        margin-bottom: 20px;
    }
    .header-meta {
        font-family: monospace;
        font-size: 12px;
        color: #8B949E;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .header-title {
        font-size: 28px;
        font-weight: 700;
        color: #F0F6FC;
        margin: 0;
    }
    .header-desc {
        font-size: 14px;
        color: #8B949E;
        margin-top: 6px;
    }
    .exec-box {
        background-color: #161B22;
        border-left: 4px solid #E34C26;
        border-top: 1px solid #30363D;
        border-right: 1px solid #30363D;
        border-bottom: 1px solid #30363D;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 20px;
    }
    .exec-title {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        color: #E34C26;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .exec-text {
        font-size: 14px;
        color: #C9D1D9;
        line-height: 1.55;
    }
    .caveat-box {
        background-color: rgba(210, 153, 34, 0.08);
        border: 1px solid rgba(210, 153, 34, 0.4);
        border-left: 4px solid #D29922;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 12px;
        margin-bottom: 16px;
        font-size: 13px;
        color: #E6EDF3;
    }
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] {
        color: #8B949E !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #F0F6FC !important;
        font-size: 22px !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 18px;
        border-bottom: 1px solid #262C36;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 4px;
        color: #8B949E;
        font-weight: 500;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        color: #F0F6FC !important;
        border-bottom: 2px solid #E34C26 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- TECHNICAL HEADER ---
st.markdown("""
<div class="header-container">
    <div class="header-meta">NASA FIRMS ARCHIVE • MULTI-SENSOR SPATIOTEMPORAL HARMONIZATION</div>
    <div class="header-title">MODIS & VIIRS Fire Activity Calendar and Radiative Energy Analysis</div>
    <div class="header-desc">Cross-sensor spatial grid alignment (0.01°), confidence calibration, month-by-month surveillance, and instantaneous radiative kinetics.</div>
</div>
""", unsafe_allow_html=True)

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
        all_items = os.listdir(base_dir)
        return None, None, None, all_items

    modis = pd.concat(modis_list, ignore_index=True)
    viirs = pd.concat(viirs_list, ignore_index=True)

    # 1. Confidence Normalization (0-100 scale)
    modis["norm_conf"] = pd.to_numeric(modis["confidence"], errors="coerce").fillna(50)
    conf_map = {"l": 30, "n": 70, "h": 95, "low": 30, "nominal": 70, "high": 95}
    viirs["norm_conf"] = viirs["confidence"].astype(str).str.lower().map(conf_map).fillna(70)

    modis = modis[modis["norm_conf"] >= 30].copy()
    viirs = viirs[viirs["norm_conf"] >= 30].copy()

    modis["sensor"] = "MODIS (1 km)"
    viirs["sensor"] = "VIIRS (375 m)"

    # 2. Spatial-Temporal Grid Alignment (0.01 deg ~ 1.11 km)
    for df in [modis, viirs]:
        df["acq_date"] = pd.to_datetime(df["acq_date"])
        df["year"] = df["acq_date"].dt.year
        df["month"] = df["acq_date"].dt.month
        df["day"] = df["acq_date"].dt.day
        df["grid_lat"] = df["latitude"].round(2)
        df["grid_lon"] = df["longitude"].round(2)

    raw_combined = pd.concat([
        modis[["latitude", "longitude", "acq_date", "year", "month", "day", "frp", "norm_conf", "sensor", "grid_lat", "grid_lon"]],
        viirs[["latitude", "longitude", "acq_date", "year", "month", "day", "frp", "norm_conf", "sensor", "grid_lat", "grid_lon"]]
    ], ignore_index=True)

    harmonized = raw_combined.groupby(["acq_date", "year", "month", "day", "grid_lat", "grid_lon"]).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        total_frp=("frp", "mean"),
        max_conf=("norm_conf", "max"),
        sensor_count=("sensor", "nunique"),
        raw_detections=("frp", "count")
    ).reset_index()

    # Composite Fire Activity Index & Instantaneous Biomass Combustion Rate (Wooster et al., 2005: 0.368 ± 0.05 kg/MJ)
    harmonized["fire_activity_index"] = harmonized["total_frp"] * (harmonized["max_conf"] / 100.0) * (1 + 0.25 * (harmonized["sensor_count"] - 1))
    harmonized["biomass_rate_kg_s"] = harmonized["total_frp"] * 0.368

    return modis, viirs, harmonized, []

modis_df, viirs_df, harm_df, debug_files = load_and_harmonize()

if harm_df is None:
    st.error(f"Dataset files not found in directory: {debug_files}")
    st.stop()

month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
full_month_names = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June", 7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"}

# --- SIDEBAR PARAMETERS & GRANULAR AOI SELECTION ---
with st.sidebar:
    st.markdown("#### Query Parameters")
    years = sorted(harm_df["year"].unique())
    selected_years = st.multiselect("Observation Period (Years)", years, default=years)

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
        st.caption(f"Bounds: {lat_range[0]}°N–{lat_range[1]}°N, {lon_range[0]}°E–{lon_range[1]}°E")

    st.markdown("---")
    st.markdown("#### Data Export")
    st.caption("Export harmonized grid records (CSV) for external GIS or CFD boundary condition setup.")

f_harm = harm_df[(harm_df["year"].isin(selected_years)) & (harm_df["latitude"].between(lat_range[0], lat_range[1])) & (harm_df["longitude"].between(lon_range[0], lon_range[1]))]
f_modis = modis_df[(modis_df["year"].isin(selected_years)) & (modis_df["latitude"].between(lat_range[0], lat_range[1])) & (modis_df["longitude"].between(lon_range[0], lon_range[1]))]
f_viirs = viirs_df[(viirs_df["year"].isin(selected_years)) & (viirs_df["latitude"].between(lat_range[0], lat_range[1])) & (viirs_df["longitude"].between(lon_range[0], lon_range[1]))]

with st.sidebar:
    csv_export = f_harm.head(10000).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Harmonized CSV",
        data=csv_export,
        file_name="harmonized_fire_events.csv",
        mime="text/csv"
    )
    st.markdown("---")
    st.caption("Data Source: NASA FIRMS (MODIS Collection 6.1 & VIIRS S-NPP C2)")
    st.caption("Radiative Uncertainty: ±15% sensor FRP, ±13.6% Wooster factor")

# --- OPERATIONAL DECISION BRIEF & SEASONAL PRIORITIZATION ---
if len(f_harm) > 0:
    monthly_counts = f_harm.groupby("month")["fire_activity_index"].count().sort_values(ascending=False)
    top_months = [month_names.get(m, str(m)) for m in monthly_counts.head(2).index]
    peak_window = " – ".join(top_months)
    primary_month = monthly_counts.index[0]

    if primary_month in [6, 7, 8, 9]:
        regime_label = "High-Intensity Canopy & Wildland Fire Regime"
        action_rec = "Prioritize pre-positioning of aerial suppression assets, thermal watchtowers, and rapid initial-attack crews 3 weeks prior to peak window."
    elif primary_month in [10, 11]:
        regime_label = "Post-Harvest Agricultural Stubble & Residue Burning Regime"
        action_rec = "Deploy ground-based agricultural inspection patrols and enforce controlled-burn permits to prevent wildland-urban interface escape."
