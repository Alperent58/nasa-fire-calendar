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
        line-height: 1.5;
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
        font-size: 23px !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
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
    <div class="header-desc">Cross-sensor spatial grid alignment (0.01°), confidence calibration, climatological anomaly detection, and bounded thermodynamic modeling.</div>
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
        df["day_of_year"] = df["acq_date"].dt.dayofyear
        df["grid_lat"] = df["latitude"].round(2)
        df["grid_lon"] = df["longitude"].round(2)

    raw_combined = pd.concat([
        modis[["latitude", "longitude", "acq_date", "year", "month", "day_of_year", "frp", "norm_conf", "sensor", "grid_lat", "grid_lon"]],
        viirs[["latitude", "longitude", "acq_date", "year", "month", "day_of_year", "frp", "norm_conf", "sensor", "grid_lat", "grid_lon"]]
    ], ignore_index=True)

    harmonized = raw_combined.groupby(["acq_date", "year", "month", "day_of_year", "grid_lat", "grid_lon"]).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        total_frp=("frp", "mean"),
        max_conf=("norm_conf", "max"),
        sensor_count=("sensor", "nunique"),
        raw_detections=("frp", "count")
    ).reset_index()

    # Composite Fire Activity Index & Biomass Consumption Rate (Wooster et al., 2005: 0.368 ± 0.05 kg/MJ)
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
    st.caption("Radiative Uncertainty: ±15% sensor FRP, ±13.6% Wooster factor")

month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}

# --- EXECUTIVE DECISION BRIEF (NON-TECHNICAL SUMMARY) ---
if len(f_harm) > 0:
    peak_month_num = f_harm.groupby("month")["fire_activity_index"].count().idxmax()
    peak_month_name = month_names.get(peak_month_num, str(peak_month_num))
    regime_note = (
        "High-intensity forest canopy & coastal wildfire regime requiring aerial suppression readiness."
        if peak_month_num in [6, 7, 8, 9]
        else "Post-harvest agricultural residue & controlled burn regime requiring ground patrol monitoring."
    )
    st.markdown(f"""
    <div class="exec-box">
        <div class="exec-title">Operational Decision Brief • {region_preset}</div>
        <div class="exec-text">
            <strong>Critical Intervention Window:</strong> Peak thermal activity occurs in <strong>{peak_month_name}</strong>. 
            <strong>Dominant Regime:</strong> {regime_note} 
            <strong>Verification Confidence:</strong> {len(f_harm[f_harm['sensor_count'] == 2]):,} events independently confirmed by both orbital sensors.
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- SUMMARY METRICS ---
col1, col2, col3, col4 = st.columns(4)
raw_total = len(f_modis) + len(f_viirs)
harm_total = len(f_harm)
reduction = ((raw_total - harm_total) / raw_total * 100) if raw_total > 0 else 0

col1.metric("Raw Sensor Detections", f"{raw_total:,}")
col2.metric("Harmonized Fire Events", f"{harm_total:,}", f"-{reduction:.1f}% spatial overlap")
col3.metric("Dual-Sensor Coincident", f"{len(f_harm[f_harm['sensor_count'] == 2]):,}")
col4.metric("Mean Radiative Power", f"{f_harm['total_frp'].mean():.1f} ± {f_harm['total_frp'].mean()*0.15:.1f} MW" if harm_total > 0 else "0.0 MW")

st.markdown("<br>", unsafe_allow_html=True)

# --- NAVIGATION TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Seasonal Calendar & Gaps",
    "Climatological Anomaly",
    "Spatial Distribution",
    "Calibration Validation",
    "Thermodynamic Modeling",
    "Methodology & Uncertainty"
])

with tab1:
    st.markdown("##### Monthly Fire Regime Matrix & Observation Coverage")
    # Check missing observation days across full calendar span
    if len(f_harm) > 0:
        min_d, max_d = f_harm["acq_date"].min(), f_harm["acq_date"].max()
        full_dates = pd.date_range(min_d, max_d, freq="D")
        observed_dates = set(f_harm["acq_date"].unique())
        missing_days = len([d for d in full_dates if d not in observed_dates])
        coverage_pct = ((len(full_dates) - missing_days) / len(full_dates)) * 100

        c_gap1, c_gap2, c_gap3 = st.columns(3)
        c_gap1.metric("Total Calendar Days Evaluated", f"{len(full_dates):,} days")
        c_gap2.metric("Days Without Active Detections / Cloud-Masked", f"{missing_days:,} days")
        c_gap3.metric("Temporal Detection Coverage", f"{coverage_pct:.1f}%")

    cal_data = f_harm.groupby(["year", "month"]).agg(fire_count=("fire_activity_index", "count")).reset_index()
    # Build complete year x month grid so missing months explicitly show as NaN/0
    all_grid = pd.MultiIndex.from_product([selected_years, range(1, 13)], names=["year", "month"]).to_frame(index=False)
    cal_full = all_grid.merge(cal_data, on=["year", "month"], how="left")
    pivot_cal = cal_full.pivot(index="year", columns="month", values="fire_count")
    pivot_cal.columns = [month_names.get(c, c) for c in pivot_cal.columns]

    fig_cal = px.imshow(
        pivot_cal,
        labels=dict(x="Month", y="Year", color="Events (Blank = No Obs)"),
        color_continuous_scale="YlOrRd",
        text_auto=True,
        aspect="auto"
    )
    fig_cal.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22", margin=dict(t=20))
    st.plotly_chart(fig_cal, use_container_width=True)
    st.caption("Note: Blank cells indicate months with zero valid orbital detections (due to persistent cloud cover, winter snowpack, or absence of thermal anomalies).")

with tab2:
    st.markdown("##### Climatological Seasonal Baseline vs. Daily Anomalies (+2σ)")
    daily = f_harm.groupby(["acq_date", "month"]).agg(event_count=("fire_activity_index", "count")).reset_index()
    # Calculate monthly climatological baseline (mean & std per calendar month)
    monthly_clim = daily.groupby("month")["event_count"].agg(clim_mean="mean", clim_std="std").reset_index()
    monthly_clim["clim_std"] = monthly_clim["clim_std"].fillna(0)
    daily = daily.merge(monthly_clim, on="month", how="left")
    daily["upper_2sigma"] = daily["clim_mean"] + 2 * daily["clim_std"]
    daily["Classification"] = np.where(daily["event_count"] > daily["upper_2sigma"], "Anomaly (> Seasonal +2σ)", "Within Seasonal Normal")
    daily = daily.sort_values("acq_date")

    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(
        x=daily["acq_date"], y=daily["clim_mean"],
        mode="lines", name="Seasonal Climatological Normal (μ)",
        line=dict(color="#58A6FF", width=2)
    ))
    fig_anom.add_trace(go.Scatter(
        x=daily["acq_date"], y=daily["upper_2sigma"],
        mode="lines", name="Seasonal +2σ Threshold",
        line=dict(color="#D29922", width=1.5, dash="dash")
    ))
    normal_pts = daily[daily["Classification"] == "Within Seasonal Normal"]
    anom_pts = daily[daily["Classification"] == "Anomaly (> Seasonal +2σ)"]

    fig_anom.add_trace(go.Scatter(
        x=normal_pts["acq_date"], y=normal_pts["event_count"],
        mode="markers", name="Normal Activity",
        marker=dict(color="#8B949E", size=5, opacity=0.6)
    ))
    fig_anom.add_trace(go.Scatter(
        x=anom_pts["acq_date"], y=anom_pts["event_count"],
        mode="markers", name="Extreme Anomaly (> +2σ)",
        marker=dict(color="#E34C26", size=8)
    ))
    fig_anom.update_layout(
        xaxis_title="Date", yaxis_title="Daily Harmonized Fire Events",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22", margin=dict(t=20)
    )
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
    st.markdown("##### Statistical Calibration Consistency & Resolution Bias Correction")
    m_monthly = f_modis.groupby("month").size().reset_index(name="MODIS (1 km)")
    v_monthly = f_viirs.groupby("month").size().reset_index(name="VIIRS (375 m)")
    h_monthly = f_harm.groupby("month").size().reset_index(name="Harmonized")
    comp = m_monthly.merge(v_monthly, on="month", how="outer").merge(h_monthly, on="month", how="outer").fillna(0)
    comp["Month"] = comp["month"].map(month_names)

    # Statistical consistency metrics between MODIS and VIIRS monthly series
    if len(comp) > 1 and comp["MODIS (1 km)"].std() > 0 and comp["VIIRS (375 m)"].std() > 0:
        r_val = np.corrcoef(comp["MODIS (1 km)"], comp["VIIRS (375 m)"])[0, 1]
        r2_val = r_val ** 2
    else:
        r_val, r2_val = 0.0, 0.0

    # FRP calibration consistency on dual-sensor coincident cells
    dual_cells = f_harm[f_harm["sensor_count"] == 2]
    dual_share = (len(dual_cells) / len(f_harm) * 100) if len(f_harm) > 0 else 0

    s1, s2, s3 = st.columns(3)
    s1.metric("Cross-Sensor Temporal Correlation (r)", f"{r_val:.3f}")
    s2.metric("Coefficient of Determination (R²)", f"{r2_val:.3f}")
    s3.metric("Coincident Grid Overlap Rate", f"{dual_share:.1f}%")

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(x=comp["Month"], y=comp["MODIS (1 km)"], name="MODIS (1 km)", marker_color="#388BFD"))
    fig_comp.add_trace(go.Bar(x=comp["Month"], y=comp["VIIRS (375 m)"], name="VIIRS (375 m)", marker_color="#D29922"))
    fig_comp.add_trace(go.Scatter(x=comp["Month"], y=comp["Harmonized"], name="Harmonized Grid Events", mode="lines+markers", line=dict(color="#E34C26", width=3)))
    fig_comp.update_layout(barmode="group", xaxis_title="Month", yaxis_title="Event Count", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22", margin=dict(t=20))
    st.plotly_chart(fig_comp, use_container_width=True)

with tab5:
    st.markdown("##### Bounded Thermodynamic Energy & Biomass Combustion Kinetics")
    st.caption("Derived from Fire Radiative Power (FRP ±15%) using Wooster et al. (2005) combustion factor (0.368 ± 0.05 kg/MJ) over a 0.01° (~1.21 km²) grid cell.")
    m1, m2, m3 = st.columns(3)
    total_mw = f_harm["total_frp"].sum()
    est_biomass_tons = (f_harm["biomass_rate_kg_s"].sum() * 3600) / 1000.0
    # Combined propagated uncertainty (~20.2% RSS from 15% FRP + 13.6% Wooster coeff)
    biomass_err = est_biomass_tons * 0.202
    peak_flux = (f_harm["total_frp"].max() * 1000) / (1100 * 1100) if harm_total > 0 else 0
    flux_err = peak_flux * 0.15

    m1.metric("Cumulative Radiative Power", f"{total_mw:,.0f} ± {total_mw*0.15:,.0f} MW")
    m2.metric("Est. Biomass Consumption (1-hr)", f"{est_biomass_tons:,.0f} ± {biomass_err:,.0f} t")
    m3.metric("Peak Grid Heat Flux", f"{peak_flux:.2f} ± {flux_err:.2f} kW/m²")
    
    monthly_energy = f_harm.groupby("month")["total_frp"].sum().reset_index()
    monthly_energy["Month"] = monthly_energy["month"].map(month_names)
    monthly_energy["frp_err"] = monthly_energy["total_frp"] * 0.15

    fig_eng = go.Figure()
    fig_eng.add_trace(go.Bar(
        x=monthly_energy["Month"],
        y=monthly_energy["total_frp"],
        error_y=dict(type="data", array=monthly_energy["frp_err"], visible=True, color="#8B949E"),
        marker_color="#E34C26",
        name="Cumulative FRP (MW)"
    ))
    fig_eng.update_layout(
        xaxis_title="Month",
        yaxis_title="Cumulative Radiative Power (MW ± 15%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#161B22",
        margin=dict(t=20)
    )
    st.plotly_chart(fig_eng, use_container_width=True)

with tab6:
    st.markdown("##### Harmonization Pipeline, Governing Equations & Uncertainty Budget")
    st.markdown("""
    1. **Spatial Resampling:** Native VIIRS (375 m) and MODIS (1 km) centroids are binned onto a unified $0.01^\\circ \\times 0.01^\\circ$ (~$1.11\\text{ km}$) geographic grid per UTC acquisition date to eliminate sub-pixel overcounting.
    2. **Confidence Calibration:** Categorical VIIRS flags (`low`, `nominal`, `high`) are mapped to quantitative equivalence scores (`30`, `70`, `95`) matching the MODIS `0–100%` scale; detections below `30%` are filtered as solar-glint / bare-soil noise per Giglio et al. (2016).
    3. **Composite Fire Activity Index (FAI):**
       $$\\text{FAI} = \\overline{\\text{FRP}} \\times \\left(\\frac{C_{\\max}}{100}\\right) \\times \\left[1 + 0.25(N_{\\text{sensors}} - 1)\\right]$$
    4. **Propagated Thermodynamic Uncertainty:**
       - Sensor FRP radiometric uncertainty: $\\delta_{\\text{FRP}} = \\pm 15\\%$
       - Wooster biomass combustion factor: $k_b = 0.368 \\pm 0.05\\ \\text{kg/MJ}$ ($\\delta_k = \\pm 13.6\\%$)
       - Root-Sum-Square (RSS) biomass consumption uncertainty: $\\delta_{\\dot{m}} = \\sqrt{0.15^2 + 0.136^2} \\approx \\pm 20.2\\%$
    """)
    st.markdown("##### Sample Harmonized Records")
    st.dataframe(f_harm.head(50), use_container_width=True)
