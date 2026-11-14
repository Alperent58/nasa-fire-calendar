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
    else:
        regime_label = "Low-Intensity / Transitional Thermal Regime"
        action_rec = "Maintain baseline satellite monitoring and verify isolated thermal detections against industrial or cloud-edge false positives."

    st.markdown(f"""
    <div class="exec-box">
        <div class="exec-title">Operational Decision Brief & Seasonal Prioritization • {region_preset}</div>
        <div class="exec-text">
            <strong>Highest-Risk Months:</strong> {peak_window} &nbsp;|&nbsp; 
            <strong>Dominant Fire Regime:</strong> {regime_label}<br>
            <strong>Preparedness Recommendation:</strong> {action_rec}
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Seasonal Calendar & Gaps",
    "Monthly Monitoring",
    "Climatological Anomaly",
    "Spatial Distribution",
    "Calibration Validation",
    "Thermodynamic Modeling",
    "Methodology & Caveats"
])

with tab1:
    st.markdown("##### Monthly Fire Regime Matrix & Observation-Gap Transparency")
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

    cal_data = f_harm.groupby(["year", "month"]).agg(fire_count=("fire_activity_index", "count")).reset_index()
    all_grid = pd.MultiIndex.from_product([selected_years, range(1, 13)], names=["year", "month"]).to_frame(index=False)
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#21262D",
        margin=dict(t=20)
    )
    st.plotly_chart(fig_cal, use_container_width=True)
    st.markdown("""
    <div class="caveat-box">
        <strong>Observation-Gap Transparency ("Empty Cell" Distinction):</strong> Dark gray blank cells represent periods with <strong>zero valid orbital detections</strong>—which may result from heavy cloud cover, dense smoke obscuration, or orbital swath gaps rather than confirmed absence of fire.
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("##### Dedicated Month-by-Month Surveillance & Daily Progression")
    default_idx = int(f_harm.groupby("month")["fire_activity_index"].count().idxmax() - 1) if len(f_harm) > 0 else 7
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
    m_frp_sum = m_harm["total_frp"].sum() if m_events > 0 else 0.0
    m_mean_rate = m_harm["biomass_rate_kg_s"].mean() if m_events > 0 else 0.0
    if m_events > 0:
        peak_day_row = m_harm.groupby("acq_date")["fire_activity_index"].count().idxmax()
        peak_day_str = peak_day_row.strftime("%Y-%m-%d")
        peak_day_cnt = m_harm.groupby("acq_date")["fire_activity_index"].count().max()
    else:
        peak_day_str, peak_day_cnt = "N/A", 0

    mc1.metric(f"{month_names[sel_month]} Harmonized Events", f"{m_events:,}", f"Raw: {len(m_modis)+len(m_viirs):,}")
    mc2.metric("Peak Activity Date", peak_day_str, f"{peak_day_cnt:,} events" if m_events > 0 else None)
    mc3.metric(f"{month_names[sel_month]} Cumulative FRP", f"{m_frp_sum:,.0f} ± {m_frp_sum*0.15:,.0f} MW")
    mc4.metric(f"{month_names[sel_month]} Mean Burn Rate", f"{m_mean_rate:.2f} ± {m_mean_rate*0.202:.2f} kg/s")

    days_in_m = calendar.monthrange(2024, sel_month)[1]
    day_grid = pd.MultiIndex.from_product([selected_years, range(1, days_in_m + 1)], names=["year", "day"]).to_frame(index=False)
    daily_m_counts = m_harm.groupby(["year", "day"]).agg(events=("fire_activity_index", "count")).reset_index()
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
        title=f"Daily Fire Regime Matrix — {full_month_names[sel_month]} (Blank = No Detection / Cloud-Masked)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#21262D",
        margin=dict(t=40)
    )
    st.plotly_chart(fig_day_cal, use_container_width=True)

    c_left, c_right = st.columns([1, 1])
    with c_left:
        yr_comp = m_harm.groupby("year").agg(
            events=("fire_activity_index", "count"),
            mean_frp=("total_frp", "mean")
        ).reindex(selected_years, fill_value=0).reset_index()

        fig_yr_m = go.Figure()
        fig_yr_m.add_trace(go.Bar(
            x=yr_comp["year"].astype(str),
            y=yr_comp["events"],
            marker_color="#E34C26",
            name="Harmonized Events"
        ))
        if len(yr_comp) > 1 and yr_comp["events"].std() > 0:
            m_thresh = yr_comp["events"].mean() + 2 * yr_comp["events"].std()
            fig_yr_m.add_hline(y=m_thresh, line_dash="dash", line_color="#D29922", annotation_text="Month +2σ Threshold")
        fig_yr_m.update_layout(
            title=f"Inter-Annual Comparison ({full_month_names[sel_month]})",
            xaxis_title="Year",
            yaxis_title="Harmonized Events",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#161B22",
            height=400,
            margin=dict(t=40)
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
                color_continuous_scale=["#D29922", "#E34C26"],
                labels={"sensor_count": "Sensors", "total_frp": "FRP (MW)"},
                hover_data=["acq_date", "total_frp", "max_conf"],
                zoom=4.8,
                height=400,
                map_style="carto-darkmatter"
            )
            fig_m_map.update_layout(
                title=f"Top Spatial Hotspots ({full_month_names[sel_month]})",
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_m_map, use_container_width=True)
        else:
            st.info(f"No valid detections recorded for {full_month_names[sel_month]} in the selected AOI and years.")

    if m_events > 0:
        m_csv = m_harm.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {full_month_names[sel_month]} Harmonized Dataset (CSV)",
            data=m_csv,
            file_name=f"harmonized_fire_{full_month_names[sel_month].lower()}.csv",
            mime="text/csv"
        )

with tab3:
    st.markdown("##### Climatological Seasonal Baseline, Anomalies (+2σ) & Missing-Observation Days")
    if len(f_harm) > 0:
        min_d, max_d = f_harm["acq_date"].min(), f_harm["acq_date"].max()
        full_calendar = pd.DataFrame({"acq_date": pd.date_range(min_d, max_d, freq="D")})
        full_calendar["month"] = full_calendar["acq_date"].dt.month

        daily_obs = f_harm.groupby("acq_date").agg(event_count=("fire_activity_index", "count")).reset_index()
        daily = full_calendar.merge(daily_obs, on="acq_date", how="left")

        monthly_clim = daily.dropna(subset=["event_count"]).groupby("month")["event_count"].agg(clim_mean="mean", clim_std="std").reset_index()
        monthly_clim["clim_std"] = monthly_clim["clim_std"].fillna(0)
        daily = daily.merge(monthly_clim, on="month", how="left")
        daily["upper_2sigma"] = daily["clim_mean"] + 2 * daily["clim_std"]

        no_obs_days = daily[daily["event_count"].isna()].copy()
        valid_days = daily.dropna(subset=["event_count"]).copy()
        valid_days["Classification"] = np.where(valid_days["event_count"] > valid_days["upper_2sigma"], "Extreme Anomaly (> Seasonal +2σ)", "Within Seasonal Normal")

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

        normal_pts = valid_days[valid_days["Classification"] == "Within Seasonal Normal"]
        anom_pts = valid_days[valid_days["Classification"] == "Extreme Anomaly (> Seasonal +2σ)"]

        fig_anom.add_trace(go.Scatter(
            x=normal_pts["acq_date"], y=normal_pts["event_count"],
            mode="markers", name="Normal Activity",
            marker=dict(color="#8B949E", size=5, opacity=0.65)
        ))
        fig_anom.add_trace(go.Scatter(
            x=anom_pts["acq_date"], y=anom_pts["event_count"],
            mode="markers", name="Extreme Anomaly (> +2σ)",
            marker=dict(color="#E34C26", size=8)
        ))
        fig_anom.add_trace(go.Scatter(
            x=no_obs_days["acq_date"], y=[0] * len(no_obs_days),
            mode="markers", name="No Observation / Cloud-Masked Day",
            marker=dict(color="#484F58", size=5, symbol="x")
        ))

        fig_anom.update_layout(
            xaxis_title="Date", yaxis_title="Daily Harmonized Fire Events",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22", margin=dict(t=20)
        )
        st.plotly_chart(fig_anom, use_container_width=True)

with tab4:
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
        height=550,
        map_style="carto-darkmatter"
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("Hotspot ≠ Burned Area: Plotted coordinates indicate active radiative thermal anomalies within 0.01° grid cells and do not directly represent total burned acreage.")

with tab5:
    st.markdown("##### Statistical Calibration Consistency & Resolution Bias Correction")
    m_monthly = f_modis.groupby("month").size().reset_index(name="MODIS (1 km)")
    v_monthly = f_viirs.groupby("month").size().reset_index(name="VIIRS (375 m)")
    h_monthly = f_harm.groupby("month").size().reset_index(name="Harmonized")
    comp = m_monthly.merge(v_monthly, on="month", how="outer").merge(h_monthly, on="month", how="outer").fillna(0)
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
    fig_comp.add_trace(go.Bar(x=comp["Month"], y=comp["MODIS (1 km)"], name="MODIS (1 km)", marker_color="#388BFD"))
    fig_comp.add_trace(go.Bar(x=comp["Month"], y=comp["VIIRS (375 m)"], name="VIIRS (375 m)", marker_color="#D29922"))
    fig_comp.add_trace(go.Scatter(x=comp["Month"], y=comp["Harmonized"], name="Harmonized Grid Events", mode="lines+markers", line=dict(color="#E34C26", width=3)))
    fig_comp.update_layout(barmode="group", xaxis_title="Month", yaxis_title="Event Count", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#161B22", margin=dict(t=20))
    st.plotly_chart(fig_comp, use_container_width=True)

with tab6:
    st.markdown("##### Bounded Radiative Power & Combustion Kinetics")
    st.caption("Instantaneous quantities derived from orbital Fire Radiative Power (FRP ±15%) and Wooster et al. (2005) combustion factor (0.368 ± 0.05 kg/MJ). Single-year selection enables annual dry biomass integration (Δt_eff = 1 hr).")
    
    total_mw = f_harm["total_frp"].sum()
    mean_burn_rate = f_harm["biomass_rate_kg_s"].mean() if harm_total > 0 else 0.0
    burn_rate_err = mean_burn_rate * 0.202
    peak_flux = (f_harm["total_frp"].max() * 1000) / (1100 * 1100) if harm_total > 0 else 0
    flux_err = peak_flux * 0.15
st.markdown("""
    <div style="background-color: #161B22; border: 1px solid #30363D; border-left: 4px solid #58A6FF; border-radius: 6px; padding: 16px 20px; margin-top: 16px;">
        <div style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: #58A6FF; letter-spacing: 0.6px; margin-bottom: 8px;">
            Governing Thermodynamic & Biomass Equations
        </div>
        <div style="font-size: 13.5px; color: #C9D1D9; line-height: 1.6;">
            • <strong>Instantaneous Biomass Burn Rate (Wooster et al., 2005):</strong> &nbsp; <code>ṁ = 0.368 × FRP [kg/s]</code> &nbsp; <em>(RSS Uncertainty: ±20.2%)</em><br>
            • <strong>Peak Grid Radiative Heat Flux (0.01° ≈ 1.21 km² cell):</strong> &nbsp; <code>q'' = (FRP_max × 1000) / (1100 × 1100) [kW/m²]</code> &nbsp; <em>(Uncertainty: ±15.0%)</em><br>
            • <strong>Single-Year Integrated Dry Biomass Loss (Δt_eff = 3600 s):</strong> &nbsp; <code>M_y = (0.368 × 3600 / 1000) × Σ FRP_i = 1.3248 × Σ FRP_i [t/yr]</code> &nbsp; <em>(RSS Uncertainty: ±32.2%)</em>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Show 4th metric card ONLY when a single year is selected
    if len(selected_years) == 1 and harm_total > 0:
        sel_yr = selected_years[0]
        annual_biomass_tons = 1.3248 * total_mw  # (0.368 * 3600 / 1000) * sum(FRP)
        annual_biomass_err = annual_biomass_tons * 0.322  # RSS: sqrt(0.15^2 + 0.136^2 + 0.25^2)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cumulative Snapshot FRP", f"{total_mw:,.0f} ± {total_mw*0.15:,.0f} MW")
        m2.metric("Mean Event Burn Rate (ṁ)", f"{mean_burn_rate:.2f} ± {burn_rate_err:.2f} kg/s")
        m3.metric("Peak Grid Heat Flux", f"{peak_flux:.2f} ± {flux_err:.2f} kW/m²")
        m4.metric(f"{sel_yr} Est. Total Biomass (M_y)", f"{annual_biomass_tons:,.0f} ± {annual_biomass_err:,.0f} t")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Cumulative Snapshot FRP", f"{total_mw:,.0f} ± {total_mw*0.15:,.0f} MW")
m2.metric("Mean Event Burn Rate (ṁ)", f"{mean_burn_rate:.2f} ± {burn_rate_err:.2f} kg/s")
