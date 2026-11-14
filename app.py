import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import os

st.set_page_config(page_title="NASA Space Apps | Fire Activity Calendar", layout="wide", page_icon="🔥")

st.title("🔥 MODIS & VIIRS Harmonized Fire Activity Calendar")
st.markdown("**NASA Space Apps Challenge:** Multi-sensor satellite data harmonization (spatial, temporal, and thermodynamic) for identifying critical wildfire regimes and anomalies.")

@st.cache_data
def load_and_harmonize():
    files = glob.glob("*.csv")
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
        return None, None, None

    modis = pd.concat(modis_list, ignore_index=True)
    viirs = pd.concat(viirs_list, ignore_index=True)

    # 1. Confidence Score Normalization (0-100 unified scale)
    modis["norm_conf"] = pd.to_numeric(modis["confidence"], errors="coerce").fillna(50)
    conf_map = {"l": 30, "n": 70, "h": 95, "low": 30, "nominal": 70, "high": 95}
    viirs["norm_conf"] = viirs["confidence"].astype(str).str.lower().map(conf_map).fillna(70)

    # Filter low-confidence thermal noise
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

    # Composite Fire Activity Index (FAI) & Biomass Consumption Rate (Wooster coeff: 0.368 kg/MJ)
    harmonized["fire_activity_index"] = harmonized["total_frp"] * (harmonized["max_conf"] / 100.0) * (1 + 0.25 * (harmonized["sensor_count"] - 1))
    harmonized["biomass_rate_kg_s"] = harmonized["total_frp"] * 0.368

    return modis, viirs, harmonized

modis_df, viirs_df, harm_df = load_and_harmonize()

if harm_df is None:
    st.error("CSV files not found! Please upload MODIS and VIIRS CSV files to the repository.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🌍 Area of Interest & Time Filter")
years = sorted(harm_df["year"].unique())
selected_years = st.sidebar.multiselect("Select Years:", years, default=years)

region_preset = st.sidebar.selectbox("Select Region Preset:", [
    "All Türkiye",
    "Mediterranean & Aegean Coast (Critical Forest Fire Zone)",
    "Southeastern Anatolia (Agricultural Stubble Burning Zone)",
    "Black Sea Region"
])

if region_preset == "Mediterranean & Aegean Coast (Critical Forest Fire Zone)":
    lat_range, lon_range = (36.0, 39.5), (26.0, 36.5)
elif region_preset == "Southeastern Anatolia (Agricultural Stubble Burning Zone)":
    lat_range, lon_range = (36.5, 38.5), (37.0, 42.5)
elif region_preset == "Black Sea Region":
    lat_range, lon_range = (40.5, 42.2), (31.0, 41.5)
else:
    lat_range, lon_range = (35.5, 42.5), (25.5, 45.0)

f_harm = harm_df[(harm_df["year"].isin(selected_years)) & (harm_df["latitude"].between(lat_range[0], lat_range[1])) & (harm_df["longitude"].between(lon_range[0], lon_range[1]))]
f_modis = modis_df[(modis_df["year"].isin(selected_years)) & (modis_df["latitude"].between(lat_range[0], lat_range[1])) & (modis_df["longitude"].between(lon_range[0], lon_range[1]))]
f_viirs = viirs_df[(viirs_df["year"].isin(selected_years)) & (viirs_df["latitude"].between(lat_range[0], lat_range[1])) & (viirs_df["longitude"].between(lon_range[0], lon_range[1]))]

# --- TOP KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
raw_total = len(f_modis) + len(f_viirs)
harm_total = len(f_harm)
reduction = ((raw_total - harm_total) / raw_total * 100) if raw_total > 0 else 0

col1.metric("Raw MODIS + VIIRS Detections", f"{raw_total:,}")
col2.metric("Harmonized Unique Fire Events", f"{harm_total:,}", f"-{reduction:.1f}% duplicates removed")
col3.metric("Dual-Sensor Verified Events", f"{len(f_harm[f_harm['sensor_count'] == 2]):,}")
col4.metric("Mean Fire Radiative Power (FRP)", f"{f_harm['total_frp'].mean():.1f} MW" if harm_total > 0 else "0 MW")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Fire Activity Calendar",
    "🚨 Unusual Conditions & Anomalies",
    "🗺️ Interactive Fire Map",
    "⚖️ Sensor Harmonization Proof",
    "⚙️ Thermodynamic & Biomass Analysis"
])

month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}

with tab1:
    st.subheader("Annual & Monthly Critical Fire Activity Heatmap")
    cal_data = f_harm.groupby(["year", "month"]).agg(fire_count=("fire_activity_index", "count")).reset_index()
    pivot_cal = cal_data.pivot(index="year", columns="month", values="fire_count").fillna(0)
    pivot_cal.columns = [month_names.get(c, c) for c in pivot_cal.columns]
    fig_cal = px.imshow(
        pivot_cal,
        labels=dict(x="Month", y="Year", color="Harmonized Fire Events"),
        color_continuous_scale="YlOrRd",
        text_auto=True,
        aspect="auto"
    )
    st.plotly_chart(fig_cal, use_container_width=True)

with tab2:
    st.subheader("Time-Series & Statistical Anomaly Detection (+2σ Threshold)")
    daily = f_harm.groupby("acq_date").agg(event_count=("fire_activity_index", "count")).reset_index()
    mean_val, std_val = daily["event_count"].mean(), daily["event_count"].std()
    threshold = mean_val + 2 * std_val
    daily["Status"] = np.where(daily["event_count"] > threshold, "🚨 EXTREME ANOMALY PERIOD", "Normal Seasonal Activity")
    fig_anom = px.scatter(
        daily,
        x="acq_date",
        y="event_count",
        color="Status",
        labels={"acq_date": "Acquisition Date", "event_count": "Daily Fire Events"},
        color_discrete_map={"🚨 EXTREME ANOMALY PERIOD": "red", "Normal Seasonal Activity": "orange"}
    )
    fig_anom.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Critical Anomaly Threshold (+2σ)")
    st.plotly_chart(fig_anom, use_container_width=True)

with tab3:
    st.subheader("Geospatial Distribution of Top High-Intensity Fire Hotspots")
    sample_map = f_harm.sort_values("fire_activity_index", ascending=False).head(3000)
    fig_map = px.scatter_map(
        sample_map,
        lat="latitude",
        lon="longitude",
        size="total_frp",
        color="sensor_count",
        color_continuous_scale=["#ff9900", "#ff0000"],
        labels={"sensor_count": "Verified Sensors", "total_frp": "FRP (MW)", "max_conf": "Confidence (%)"},
        hover_data=["acq_date", "total_frp", "max_conf"],
        zoom=5,
        height=550,
        map_style="open-street-map"
    )
    st.plotly_chart(fig_map, use_container_width=True)

with tab4:
    st.subheader("Pre- vs. Post-Harmonization Sensor Comparison")
    m_monthly = f_modis.groupby("month").size().reset_index(name="MODIS (Raw)")
    v_monthly = f_viirs.groupby("month").size().reset_index(name="VIIRS (Raw)")
    h_monthly = f_harm.groupby("month").size().reset_index(name="Harmonized")
    comp = m_monthly.merge(v_monthly, on="month", how="outer").merge(h_monthly, on="month", how="outer").fillna(0)
    comp["Month"] = comp["month"].map(month_names)
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(x=comp["Month"], y=comp["MODIS (Raw)"], name="MODIS (1 km - Coarse Resolution)"))
    fig_comp.add_trace(go.Bar(x=comp["Month"], y=comp["VIIRS (Raw)"], name="VIIRS (375 m - Pixel Inflation)"))
    fig_comp.add_trace(go.Scatter(x=comp["Month"], y=comp["Harmonized"], name="Harmonized Ground-Truth Events", mode="lines+markers", line=dict(color="red", width=4)))
    fig_comp.update_layout(barmode="group", xaxis_title="Month", yaxis_title="Number of Detections / Events")
    st.plotly_chart(fig_comp, use_container_width=True)

with tab5:
    st.subheader("Thermodynamic Energy & Biomass Combustion Kinetics")
    st.markdown("Physical quantities derived from satellite radiometric measurements (**FRP - MW**) via Stefan-Boltzmann radiation and Wooster combustion kinetics ($\\dot{m} = 0.368 \\times \\text{FRP}$):")
    m1, m2, m3 = st.columns(3)
    total_mw = f_harm["total_frp"].sum()
    est_biomass_tons = (f_harm["biomass_rate_kg_s"].sum() * 3600) / 1000.0
    peak_flux = (f_harm["total_frp"].max() * 1000) / (1100 * 1100) if harm_total > 0 else 0
    m1.metric("Cumulative Radiative Power", f"{total_mw:,.0f} MW")
    m2.metric("Est. Combusted Biomass (1-hr avg)", f"{est_biomass_tons:,.0f} Tons")
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
    st.plotly_chart(fig_eng, use_container_width=True)
