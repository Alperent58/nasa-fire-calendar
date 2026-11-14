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
        font-size: 24px !important;
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
    modis["norm_conf
