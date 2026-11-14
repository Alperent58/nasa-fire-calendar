import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import os

st.set_page_config(page_title="NASA Space Apps | Fire Activity Calendar", layout="wide", page_icon="🔥")

st.title("🔥 MODIS & VIIRS Uyumlaştırılmış Yangın Aktivitesi Takvimi")
st.markdown("**NASA Space Apps Challenge:** Çoklu uydu sensör verilerinin mekansal, zamansal ve termodinamik uyumlaştırılması ile kritik yangın dönemlerinin tespiti.")

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

    # 1. Güven Normalizasyonu (0-100 ölçeği)
    modis["norm_conf"] = pd.to_numeric(modis["confidence"], errors="coerce").fillna(50)
    conf_map = {"l": 30, "n": 70, "h": 95, "low": 30, "nominal": 70, "high": 95}
    viirs["norm_conf"] = viirs["confidence"].astype(str).str.lower().map(conf_map).fillna(70)

    modis = modis[modis["norm_conf"] >= 30].copy()
    viirs = viirs[viirs["norm_conf"] >= 30].copy()

    modis["sensor"] = "MODIS (1 km)"
    viirs["sensor"] = "VIIRS (375 m)"

    # 2. Mekansal Grid Uyumlaştırması (0.01 derece ~ 1.1 km)
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

    # Birleşik Yangın Aktivite İndeksi ve Biyokütle Tüketim Hızı (Wooster katsayısı: 0.368 kg/MJ)
    harmonized["fire_activity_index"] = harmonized["total_frp"] * (harmonized["max_conf"] / 100.0) * (1 + 0.25 * (harmonized["sensor_count"] - 1))
    harmonized["biomass_rate_kg_s"] = harmonized["total_frp"] * 0.368

    return modis, viirs, harmonized

modis_df, viirs_df, harm_df = load_and_harmonize()

if harm_df is None:
    st.error("CSV dosyaları bulunamadı! Lütfen GitHub deposuna MODIS ve VIIRS CSV dosyalarını yükleyin.")
    st.stop()

# --- YAN MENÜ ---
st.sidebar.header("🌍 İlgi Alanı ve Zaman Filtresi")
years = sorted(harm_df["year"].unique())
selected_years = st.sidebar.multiselect("Yılları Seçin:", years, default=years)

region_preset = st.sidebar.selectbox("Hazır Bölge Seçimi:", [
    "Tüm Türkiye",
    "Akdeniz & Ege Kıyı Şeridi (Kritik Orman Yangını Bölgesi)",
    "Güneydoğu Anadolu (Tarımsal Anız Yakma Bölgesi)",
    "Karadeniz Bölgesi"
])

if region_preset == "Akdeniz & Ege Kıyı Şeridi (Kritik Orman Yangını Bölgesi)":
    lat_range, lon_range = (36.0, 39.5), (26.0, 36.5)
elif region_preset == "Güneydoğu Anadolu (Tarımsal Anız Yakma Bölgesi)":
    lat_range, lon_range = (36.5, 38.5), (37.0, 42.5)
elif region_preset == "Karadeniz Bölgesi":
    lat_range, lon_range = (40.5, 42.2), (31.0, 41.5)
else:
    lat_range, lon_range = (35.5, 42.5), (25.5, 45.0)

f_harm = harm_df[(harm_df["year"].isin(selected_years)) & (harm_df["latitude"].between(lat_range[0], lat_range[1])) & (harm_df["longitude"].between(lon_range[0], lon_range[1]))]
f_modis = modis_df[(modis_df["year"].isin(selected_years)) & (modis_df["latitude"].between(lat_range[0], lat_range[1])) & (modis_df["longitude"].between(lon_range[0], lon_range[1]))]
f_viirs = viirs_df[(viirs_df["year"].isin(selected_years)) & (viirs_df["latitude"].between(lat_range[0], lat_range[1])) & (viirs_df["longitude"].between(lon_range[0], lon_range[1]))]

# --- ÜST METRİKLER ---
col1, col2, col3, col4 = st.columns(4)
raw_total = len(f_modis) + len(f_viirs)
harm_total = len(f_harm)
reduction = ((raw_total - harm_total) / raw_total * 100) if raw_total > 0 else 0

col1.metric("Ham MODIS + VIIRS Tespitleri", f"{raw_total:,}")
col2.metric("Uyumlaştırılmış Tekil Yangın Odağı", f"{harm_total:,}", f"-{reduction:.1f}% mükerrer temizlendi")
col3.metric("Çift Sensör Doğrulamalı Yangın", f"{len(f_harm[f_harm['sensor_count'] == 2]):,}")
col4.metric("Ortalama Yangın Gücü (FRP)", f"{f_harm['total_frp'].mean():.1f} MW" if harm_total > 0 else "0 MW")

# --- SEKMELER ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Yangın Aktivitesi Takvimi",
    "🚨 Olağandışı Koşullar (Anomali)",
    "🗺️ İnteraktif Yangın Haritası",
    "⚖️ Sensör Uyumlaştırma Kanıtı",
    "⚙️ Termodinamik & Biyokütle Analizi"
])

with tab1:
    st.subheader("Yıllık ve Aylık Kritik Yangın Dönemleri Takvimi")
    cal_data = f_harm.groupby(["year", "month"]).agg(yangin_sayisi=("fire_activity_index", "count")).reset_index()
    pivot_cal = cal_data.pivot(index="year", columns="month", values="yangin_sayisi").fillna(0)
    month_names = {1:"Oca", 2:"Şub", 3:"Mar", 4:"Nis", 5:"May", 6:"Haz", 7:"Tem", 8:"Ağu", 9:"Eyl", 10:"Eki", 11:"Kas", 12:"Ara"}
    pivot_cal.columns = [month_names.get(c, c) for c in pivot_cal.columns]
    fig_cal = px.imshow(pivot_cal, labels=dict(x="Ay", y="Yıl", color="Yangın Olayı"), color_continuous_scale="YlOrRd", text_auto=True, aspect="auto")
    st.plotly_chart(fig_cal, use_container_width=True)

with tab2:
    st.subheader("Zaman Serisi ve Olağandışı Durum (Anomali) Analizi")
    daily = f_harm.groupby("acq_date").agg(olay_sayisi=("fire_activity_index", "count")).reset_index()
    mean_val, std_val = daily["olay_sayisi"].mean(), daily["olay_sayisi"].std()
    threshold = mean_val + 2 * std_val
    daily["Durum"] = np.where(daily["olay_sayisi"] > threshold, "🚨 OLAĞANDIŞI KRİTİK DÖNEM", "Normal Aktivite")
    fig_anom = px.scatter(daily, x="acq_date", y="olay_sayisi", color="Durum", color_discrete_map={"🚨 OLAĞANDIŞI KRİTİK DÖNEM": "red", "Normal Aktivite": "orange"})
    fig_anom.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Kritik Anomali Eşiği (+2σ)")
    st.plotly_chart(fig_anom, use_container_width=True)

with tab3:
    st.subheader("Uyumlaştırılmış Kritik Noktalar Haritası")
    sample_map = f_harm.sort_values("fire_activity_index", ascending=False).head(3000)
    fig_map = px.scatter_map(sample_map, lat="latitude", lon="longitude", size="total_frp", color="sensor_count", color_continuous_scale=["#ff9900", "#ff0000"], hover_data=["acq_date", "total_frp", "max_conf"], zoom=5, height=550, map_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

with tab4:
    st.subheader("Sensör Uyumlaştırma Öncesi ve Sonrası Karşılaştırması")
    m_monthly = f_modis.groupby("month").size().reset_index(name="MODIS (Ham)")
    v_monthly = f_viirs.groupby("month").size().reset_index(name="VIIRS (Ham)")
    h_monthly = f_harm.groupby("month").size().reset_index(name="Uyumlaştırılmış")
    comp = m_monthly.merge(v_monthly, on="month", how="outer").merge(h_monthly, on="month", how="outer").fillna(0)
    comp["Ay"] = comp["month"].map({1:"Oca", 2:"Şub", 3:"Mar", 4:"Nis", 5:"May", 6:"Haz", 7:"Tem", 8:"Ağu", 9:"Eyl", 10:"Eki", 11:"Kas", 12:"Ara"})
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(x=comp["Ay"], y=comp["MODIS (Ham)"], name="MODIS (1 km)"))
    fig_comp.add_trace(go.Bar(x=comp["Ay"], y=comp["VIIRS (Ham)"], name="VIIRS (375 m)"))
    fig_comp.add_trace(go.Scatter(x=comp["Ay"], y=comp["Uyumlaştırılmış"], name="Uyumlaştırılmış Gerçek Olay", mode="lines+markers", line=dict(color="red", width=4)))
    fig_comp.update_layout(barmode="group")
    st.plotly_chart(fig_comp, use_container_width=True)

with tab5:
    st.subheader("Mühendislik Termodinamik Enerji ve Biyokütle Tüketim Analizi")
    st.markdown("Uydu radyometrik ölçümlerinden (**FRP - MW**) Stefan-Boltzmann ve Wooster yanma kinetiği modeli ($\\dot{m} = 0.368 \\times \\text{FRP}$) ile hesaplanan fiziksel büyüklükler:")
    m1, m2, m3 = st.columns(3)
    total_mw = f_harm["total_frp"].sum()
    est_biomass_tons = (f_harm["biomass_rate_kg_s"].sum() * 3600) / 1000.0
    peak_flux = (f_harm["total_frp"].max() * 1000) / (1100 * 1100) if harm_total > 0 else 0
    m1.metric("Kümülatif Radyatif Güç", f"{total_mw:,.0f} MW")
    m2.metric("Tahmini Yanan Biyokütle (1 Saatlik Ort.)", f"{est_biomass_tons:,.0f} Ton")
    m3.metric("Maksimum Birim Isı Akısı", f"{peak_flux:.2f} kW/m²")
    
    monthly_energy = f_harm.groupby("month")["total_frp"].sum().reset_index()
    monthly_energy["Ay"] = monthly_energy["month"].map({1:"Oca", 2:"Şub", 3:"Mar", 4:"Nis", 5:"May", 6:"Haz", 7:"Tem", 8:"Ağu", 9:"Eyl", 10:"Eki", 11:"Kas", 12:"Ara"})
    fig_eng = px.bar(monthly_energy, x="Ay", y="total_frp", title="Aylara Göre Toplam Termal Radyasyon Enerjisi (MW)", color="total_frp", color_continuous_scale="Inferno")
    st.plotly_chart(fig_eng, use_container_width=True)
