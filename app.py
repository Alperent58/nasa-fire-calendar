import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import os

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
        margin-bottom: 24px;
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
        font-size: 24px !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
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
    <div class="header-desc">Cross-sensor spatial grid alignment (0.01°), confidence normalization, and thermodynamic biomass combustion modeling.</div>
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

    # Composite Fire Activity Index & Biomass Consumption Rate (Wooster et al., 2005: 0.368 kg/MJ)
    harmonized["fire_activity_index"] = harmonized["total_frp"] * (harmonized["max_conf"] / 100.0) * (1 + 0.25 * (harmonized["sensor_count"] - 1))
    harmonized["biomass_rate_kg_s"] = harmonized["total_frp"] * 0.368

    return modis, viirs, harmonized, []

modis_df, viirs_df, harm_df, debug_files = load_and_harmonize()

if harm_df is None:
    st.error(f"Dataset files not found in directory: {debug_files}")
    st.stop()

# --- SIDEBAR PARAMETERS ---
with st.sidebar:
    st.markdown("#### Query Parameters")
    years = sorted(harm_df["year"].unique())
    selected_years = st.multiselect("Observation Period (Years)", years, default=years)

    region_preset = st.selectbox("Area of Interest (AOI)", [
        "All Türkiye (35.5°N–42.5°N)",
        "Mediterranean & Aegean Coast",
        "Southeastern Anatolia",
        "Black Sea Region"
    ])

    st.markdown("---")
    st.markdown("#### Data Export")
    st.caption("Export harmonized grid records (CSV) for external GIS or CFD boundary condition setup.")

if region_preset == "Mediterranean & Aegean Coast":
    lat_range, lon_range = (36.0, 39.5), (26.0, 36.5)
elif region_preset == "Southeastern Anatolia":
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
        label="Download Harmonized CSV",
        data=csv_export,
        file_name="harmonized_fire_events.csv",
        mime="text/csv"
    )
    st.markdown("---")
    st.caption("Data Source: NASA FIRMS (MODIS Collection 6.1 & VIIRS S-NPP C2)")

# --- SUMMARY METRICS ---
col1, col2, col3, col4 = st.columns(4)
raw_total = len(f_modis) + len(f_viirs)
harm_total = len(f_harm)
reduction = ((raw_total - harm_total) / raw_total * 100) if raw_total > 0 else 0

col1.metric("Raw Sensor Detections", f"{raw_total:,}")
col2.metric("Harmonized Fire Events", f"{harm_total:,}", f"-{reduction:.1f}% spatial overlap")
col3.metric("Dual-Sensor Coincident", f"{len(f_harm[f_harm['sensor_count'] == 2]):,}")
col4.metric("Mean Radiative Power", f"{f_harm['total_frp'].mean():.1f} MW" if harm_total > 0 else "0.0 MW")

st.markdown("<br>", unsafe_allow_html=True)

# --- NAVIGATION TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Seasonal Calendar",
    "Anomaly Analysis",
    "Spatial Distribution",
    "Sensor Intercomparison",
    "Thermodynamic Modeling",
    "Data & Methodology"
])

month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}

with tab1:
    st.markdown("##### Monthly Fire Regime Matrix (Harmonized Event Count)")
    cal_data = f_harm.groupby(["year", "month"]).agg(fire_count=("fire_activity_index", "count")).reset_index()
    pivot_cal = cal_data.pivot(index="year", columns="month", values="fire_count").fillna(0)
    pivot_cal.columns = [month_names.get(c, c) for c in pivot_cal.columns]
    fig_cal = px.imshow(
        pivot_cal,
        labels=dict(x="Month", y="Year", color="Events"),
        color_continuous_scale="YlOrRd",
        text_auto=True,
        aspect="auto"
    )
    fig_cal.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=20))
    st.plotly_chart(fig_cal, use_container_width=True)

with tab2:
    st.markdown("##### Daily Time-Series & Statistical Thresholding (μ + 2σ)")
    daily = f_harm.groupby("acq_date").agg(event_count=("fire_activity_index", "count")).reset_index()
    mean_val, std_val = daily["event_count"].mean(), daily["event_count"].std()
    threshold = mean_val + 2 * std_val
    daily["Classification"] = np.where(daily["event_count"] > threshold, "Anomaly (> +2σ)", "Baseline Regime")
    fig_anom = px.scatter(
        daily,
        x="acq_date",
        y="event_count",
        color="Classification",
        labels={"acq_date": "Date", "event_count": "Daily Fire Events"},
        color_discrete_map={"Anomaly (> +2σ)": "#E34C26", "Baseline Regime": "#58A6FF"}
    )
    fig_anom.add_hline(y=threshold, line_dash="dash", line_color="#E34C26", annotation_text=f"2σ Threshold ({threshold:.1f})")
    fig_anom.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22", margin=dict(t=20))
    st.plotly_chart(fig_anom, use_container_width=True)

with tab3:
    st.markdown("##### Geospatial Distribution of High-Intensity Events (Top 3,000 by FAI)")
    sample_map = f_harm.sort_values("fire_activity_index", ascending=False).head(3000)
    fig_map = px.scatter_map(
        sample_map,
        lat="latitude",
        lon="longitude",
        size="total_frp",
        color="sensor_count",
        color_continuous_scale=["#D29922", "#E34C26"],
        labels={"sensor_count": "Sensor Count", "total_frp": "FRP (MW)", "max_conf": "Confidence (%)"},
        hover_data=["acq_date", "total_frp", "max_conf"],
        zoom=5,
        height=560,
        map_style="carto-darkmatter"
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_map, use_container_width=True)

with tab4:
    st.markdown("##### Resolution Bias & Sub-Pixel Duplication Correction")
    m_monthly = f_modis.groupby("month").size().reset_index(name="MODIS (1 km)")
    v_monthly = f_viirs.groupby("month").size().reset_index(name="VIIRS (375 m)")
    h_monthly = f_harm.groupby("month").size().reset_index(name="Harmonized")
    comp = m_monthly.merge(v_monthly, on="month", how="outer").merge(h_monthly, on="month", how="outer").fillna(0)
    comp["Month"] = comp["month"].map(month_names)
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(x=comp["Month"], y=comp["MODIS (1 km)"], name="MODIS (1 km)", marker_color="#388BFD"))
    fig_comp.add_trace(go.Bar(x=comp["Month"], y=comp["VIIRS (375 m)"], name="VIIRS (375 m)", marker_color="#D29922"))
    fig_comp.add_trace(go.Scatter(x=comp["Month"], y=comp["Harmonized"], name="Harmonized Grid Events", mode="lines+markers", line=dict(color="#E34C26", width=3)))
    fig_comp.update_layout(barmode="group", xaxis_title="Month", yaxis_title="Event Count", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22", margin=dict(t=20))
    st.plotly_chart(fig_comp, use_container_width=True)

with tab5:
    st.markdown("##### Radiative Energy Balance & Biomass Combustion Kinetics")
    st.caption("Derived from Fire Radiative Power (FRP) using Wooster et al. (2005) combustion factor (0.368 kg/MJ) over a 0.01° (~1.21 km²) grid cell.")
    m1, m2, m3 = st.columns(3)
    total_mw = f_harm["total_frp"].sum()
    est_biomass_tons = (f_harm["biomass_rate_kg_s"].sum() * 3600) / 1000.0
    peak_flux = (f_harm["total_frp"].max() * 1000) / (1100 * 1100) if harm_total > 0 else 0
    m1.metric("Cumulative Radiative Power", f"{total_mw:,.0f} MW")
    m2.metric("Est. Biomass Consumption (1-hr)", f"{est_biomass_tons:,.0f} t")
    m3.metric("Peak Grid Heat Flux", f"{peak_flux:.2f} kW/m²")
    
    monthly_energy = f_harm.groupby("month")["total_frp"].sum().reset_index()
    monthly_energy["Month"] = monthly_energy["month"].map(month_names)
    fig_eng = px.bar(
        monthly_energy,
        x="Month",
        y="total_frp",
        labels={"total_frp": "Cumulative FRP (MW)"},
        color="total_frp",
        color_continuous_scale="YlOrRd"
    )
    fig_eng.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22", margin=dict(t=20))
    st.plotly_chart(fig_eng, use_container_width=True)

with tab6:
    st.markdown("##### Harmonization Pipeline & Governing Equations")
    st.markdown("""
    1. **Spatial Resampling:** Native VIIRS (375 m) and MODIS (1 km) centroids are binned onto a unified $0.01^\\circ \\times 0.01^\\circ$ (~$1.11\\text{ km}$) geographic grid per UTC acquisition date to prevent sub-pixel overcounting.
    2. **Confidence Normalization:** Categorical VIIRS flags (`low`, `nominal`, `high`) are mapped to quantitative equivalence scores (`30`, `70`, `95`) matching the MODIS `0–100%` scale; detections below `30%` are filtered as thermal noise.
    3. **Composite Fire Activity Index (FAI):**
       $$\\text{FAI} = \\overline{\\text{FRP}} \\times \\left(\\frac{C_{\\max}}{100}\\right) \\times \\left[1 + 0.25(N_{\\text{sensors}} - 1)\\right]$$
    4. **Biomass Combustion Rate:** Estimated via radiative energy balance $\\dot{m}_b = 0.368 \\times \\text{FRP}\\ [\\text{kg/s}]$.
    """)
    st.markdown("##### Sample Harmonized Records")
    st.dataframe(f_harm.head(50), use_container_width=True)
