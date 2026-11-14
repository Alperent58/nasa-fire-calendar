import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import os

st.set_page_config(
    page_title="NASA Space Apps | Fire Activity Calendar",
    layout="wide",
    page_icon="🛰️",
    initial_sidebar_state="expanded"
)

# --- CUSTOM NASA MISSION CONTROL CSS ---
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top right, #162038 0%, #0B0F19 60%);
    }
    .nasa-banner {
        background: linear-gradient(90deg, rgba(255,75,43,0.15) 0%, rgba(19,26,42,0.9) 100%);
        border-left: 5px solid #FF4B2B;
        border-radius: 10px;
        padding: 20px 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .nasa-badge {
        display: inline-block;
        background-color: rgba(255, 75, 43, 0.2);
        color: #FF6B4A;
        border: 1px solid #FF4B2B;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .nasa-title {
        font-size: 32px;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .nasa-subtitle {
        font-size: 15px;
        color: #94A3B8;
        margin-top: 6px;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(19, 26, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 3px solid #FF4B2B;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(19, 26, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 18px;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B2B !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER BANNER ---
st.markdown("""
<div class="nasa-banner">
    <div class="nasa-badge">🛰️ NASA SPACE APPS CHALLENGE • MULTI-SENSOR FUSION ENGINE</div>
    <div class="nasa-title">🔥 MODIS & VIIRS Harmonized Fire Activity Calendar</div>
    <div class="nasa-subtitle">Spatial, temporal, and thermodynamic harmonization of orbital thermal anomalies for critical wildfire regime detection.</div>
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

    # 1. Confidence Score Normalization (0-100 unified scale)
    modis["norm_conf"] = pd.to_numeric(modis["confidence"], errors="coerce").fillna(50)
    conf_map = {"l": 30, "n": 70, "h": 95, "low": 30, "nominal": 70, "high": 95}
    viirs["norm_conf"] = viirs["confidence"].astype(str).str.lower().map(conf_map).fillna(70)

    modis = modis[modis["norm_conf"] >= 30].copy()
    viirs = viirs[viirs["norm_conf"] >= 30].copy()

    modis["sensor"] = "MODIS (1 km)"
    viirs["sensor"] = "VIIRS (375 m)"

    # 2. Spatial-Temporal Grid Alignment (0.01 deg ~ 1.1 km resolution)
    for df in [modis, viirs]:
        df["acq_date"] = pd.to_datetime(df["acq_date"])
        df["year"] = df["acq_date"].dt.year
        df["month"] = df["acq_date"].dt.month
        df["grid_lat"] = df["latitude"].round(2)
        df["grid_lon"] = df["longitude"].round(2)

    raw_combined = pd.concat([
        modis[["latitude", "longitude", "acq_date", "year", "month", "frp", "norm_conf", "sensor", "grid_lat", "grid_lon"]],
        viirs[["latitude", "longitude", "acq_date", "year", "month", "frp", "norm_conf", "sensor", "grid_lat", "grid_lon"]]
    ], ignore_index=True)

    harmonized = raw_combined.groupby(["acq_date", "year", "month", "grid_lat", "grid_lon"]).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        total_frp=("frp", "mean"),
        max_conf=("norm_conf", "max"),
        sensor_count=("sensor", "nunique"),
        raw_detections=("frp", "count")
    ).reset_index()

    harmonized["fire_activity_index"] = harmonized["total_frp"] * (harmonized["max_conf"] / 100.0) * (1 + 0.25 * (harmonized["sensor_count"] - 1))
    harmonized["biomass_rate_kg_s"] = harmonized["total_frp"] * 0.368

    return modis, viirs, harmonized, []

modis_df, viirs_df, harm_df, debug_files = load_and_harmonize()

if harm_df is None:
    st.error(f"CSV files not found! Files currently inside repository: {debug_files}")
    st.stop()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("### 🎛️ Mission Control Filters")
    years = sorted(harm_df["year"].unique())
    selected_years = st.multiselect("Observation Years:", years, default=years)

    region_preset = st.selectbox("Target Region (AOI):", [
        "All Türkiye",
        "Mediterranean & Aegean Coast (Forest Fire Zone)",
        "Southeastern Anatolia (Agricultural Stubble Zone)",
        "Black Sea Region"
    ])

    st.markdown("---")
    st.markdown("### 📥 Export Harmonized Data")
    st.caption("Download the deduplicated, multi-sensor harmonized dataset for external CFD/GIS modeling.")

if region_preset == "Mediterranean & Aegean Coast (Forest Fire Zone)":
    lat_range, lon_range = (36.0, 39.5), (26.0, 36.5)
elif region_preset == "Southeastern Anatolia (Agricultural Stubble Zone)":
    lat_range, lon_range = (36.5, 38.5), (37.0, 42.5)
elif region_preset == "Black Sea Region":
    lat_range, lon_range = (40.5, 42.2), (31.0, 41.5)
else:
    lat_range, lon_range = (35.5, 42.5), (25.5, 45.0)

f_harm = harm_df[(harm_df["year"].isin(selected_years)) & (harm_df["latitude"].between(lat_range[0], lat_range[1])) & (harm_df["longitude"].between(lon_range[0], lon_range[1]))]
f_modis = modis_df[(modis_df["year"].isin(selected_years)) & (modis_df["latitude"].between(lat_range[0], lat_range[1])) & (modis_df["longitude"].between(lon_range[0], lon_range[1]))]
f_viirs = viirs_df[(viirs_df["year"].isin(selected_years)) & (viirs_df["latitude"].between(lat_range[0], lat_range[1])) & (viirs_df["longitude"].between(lon_range[0], lon_range[1]))]

with st.sidebar:
    csv_export = f_harm.head(10000).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Harmonized CSV",
        data=csv_export,
        file_name="harmonized_nasa_fire_events.csv",
        mime="text/csv"
    )

# --- TOP KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
raw_total = len(f_modis) + len(f_viirs)
harm_total = len(f_harm)
reduction = ((raw_total - harm_total) / raw_total * 100) if raw_total > 0 else 0

col1.metric("Raw Sensor Detections", f"{raw_total:,}")
col2.metric("Harmonized Fire Events", f"{harm_total:,}", f"-{reduction:.1f}% duplicates merged")
col3.metric("Dual-Sensor Verified", f"{len(f_harm[f_harm['sensor_count'] == 2]):,}")
col4.metric("Mean Radiative Power", f"{f_harm['total_frp'].mean():.1f} MW" if harm_total > 0 else "0 MW")

st.markdown("<br>", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Fire Activity Calendar",
    "🚨 Anomaly Detection",
    "🗺️ Geospatial Hotspots",
    "⚖️ Sensor Harmonization",
    "⚙️ Thermodynamics & Kinetics"
])

month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}

with tab1:
    st.subheader("Annual & Monthly Critical Fire Regime Matrix")
    cal_data = f_harm.groupby(["year", "month"]).agg(fire_count=("fire_activity_index", "count")).reset_index()
    pivot_cal = cal_data.pivot(index="year", columns="month", values="fire_count").fillna(0)
    pivot_cal.columns = [month_names.get(c, c) for c in pivot_cal.columns]
    fig_cal = px.imshow(
        pivot_cal,
        labels=dict(x="Month", y="Year", color="Fire Events"),
        color_continuous_scale="Inferno",
        text_auto=True,
        aspect="auto"
    )
    fig_cal.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_cal, use_container_width=True)

with tab2:
    st.subheader("Time-Series & Extreme Anomaly Detection (+2σ Threshold)")
    daily = f_harm.groupby("acq_date").agg(event_count=("fire_activity_index", "count")).reset_index()
    mean_val, std_val = daily["event_count"].mean(), daily["event_count"].std()
    threshold = mean_val + 2 * std_val
    daily["Status"] = np.where(daily["event_count"] > threshold, "🚨 EXTREME ANOMALY", "Normal Seasonal Regime")
    fig_anom = px.scatter(
        daily,
        x="acq_date",
        y="event_count",
        color="Status",
        labels={"acq_date": "Acquisition Date", "event_count": "Daily Harmonized Events"},
        color_discrete_map={"🚨 EXTREME ANOMALY": "#FF4B2B", "Normal Seasonal Regime": "#F59E0B"}
    )
    fig_anom.add_hline(y=threshold, line_dash="dash", line_color="#FF4B2B", annotation_text="Critical Threshold (+2σ)")
    fig_anom.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,26,42,0.5)")
    st.plotly_chart(fig_anom, use_container_width=True)

with tab3:
    st.subheader("Orbital Thermal Hotspot Distribution (Top 3,000 High-Intensity Events)")
    sample_map = f_harm.sort_values("fire_activity_index", ascending=False).head(3000)
    fig_map = px.scatter_map(
        sample_map,
        lat="latitude",
        lon="longitude",
        size="total_frp",
        color="sensor_count",
        color_continuous_scale=["#F59E0B", "#FF2A00"],
        labels={"sensor_count": "Verified Sensors", "total_frp": "FRP (MW)", "max_conf": "Confidence (%)"},
        hover_data=["acq_date", "total_frp", "max_conf"],
        zoom=5,
        height=580,
        map_style="carto-darkmatter"
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_map, use_container_width=True)

with tab4:
    st.subheader("Multi-Sensor Resolution & Pixel Inflation Correction")
    m_monthly = f_modis.groupby("month").size().reset_index(name="MODIS (Raw)")
    v_monthly = f_viirs.groupby("month").size().reset_index(name="VIIRS (Raw)")
    h_monthly = f_harm.groupby("month").size().reset_index(name="Harmonized")
    comp = m_monthly.merge(v_monthly, on="month", how="outer").merge(h_monthly, on="month", how="outer").fillna(0)
    comp["Month"] = comp["month"].map(month_names)
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(x=comp["Month"], y=comp["MODIS (Raw)"], name="MODIS (1 km)", marker_color="#3B82F6"))
    fig_comp.add_trace(go.Bar(x=comp["Month"], y=comp["VIIRS (Raw)"], name="VIIRS (375 m)", marker_color="#F59E0B"))
    fig_comp.add_trace(go.Scatter(x=comp["Month"], y=comp["Harmonized"], name="Harmonized Ground-Truth", mode="lines+markers", line=dict(color="#FF4B2B", width=4)))
    fig_comp.update_layout(barmode="group", xaxis_title="Month", yaxis_title="Detections / Events", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,26,42,0.5)")
    st.plotly_chart(fig_comp, use_container_width=True)

with tab5:
    st.subheader("Thermodynamic Energy & Biomass Combustion Kinetics")
    st.markdown("Physical quantities derived from orbital radiometric measurements (**FRP - MW**) via Stefan-Boltzmann radiation and Wooster combustion kinetics ($\\dot{m} = 0.368 \\times \\text{FRP}$):")
    m1, m2, m3 = st.columns(3)
    total_mw = f_harm["total_frp"].sum()
    est_biomass_tons = (f_harm["biomass_rate_kg_s"].sum() * 3600) / 1000.0
    peak_flux = (f_harm["total_frp"].max() * 1000) / (1100 * 1100) if harm_total > 0 else 0
    m1.metric("Cumulative Radiative Power", f"{total_mw:,.0f} MW")
    m2.metric("Est. Combusted Biomass (1-hr)", f"{est_biomass_tons:,.0f} Tons")
    m3.metric("Peak Radiative Heat Flux", f"{peak_flux:.2f} kW/m²")
    
    monthly_energy = f_harm.groupby("month")["total_frp"].sum().reset_index()
    monthly_energy["Month"] = monthly_energy["month"].map(month_names)
    fig_eng = px.bar(
        monthly_energy,
        x="Month",
        y="total_frp",
        labels={"total_frp": "Total Radiative Power (MW)"},
        title="Monthly Cumulative Thermal Radiative Power (MW)",
        color="total_frp",
        color_continuous_scale="Inferno"
    )
    fig_eng.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,26,42,0.5)")
    st.plotly_chart(fig_eng, use_container_width=True)
