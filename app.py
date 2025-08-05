import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random

from markov_model import top6_markov, top6_markov_order2, top6_markov_hybrid
from ai_model import (
    top6_model,
    train_and_save_model,
    kombinasi_4d,
    evaluate_lstm_accuracy_all_digits,
    preprocess_data,
    find_best_window_size_with_model_true,
    build_lstm_model,
    build_transformer_model
)
from lokasi_list import lokasi_list
from user_manual import tampilkan_user_manual
from ws_scan_catboost import (
    scan_ws_catboost,
    train_temp_lstm_model,
    get_top6_lstm_temp,
    show_catboost_heatmaps
)
from tab3 import tab3
from tab4 import tab4
from tab5 import tab5
from tab6 import tab6

st.set_page_config(page_title="Prediksi AI", layout="wide")

st.title("Prediksi 4D - AI")

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

def generate_dynamic_settings(df):
    if df.empty or len(df) < 10:
        return {}

    settings = {}
    data_size = len(df)
    numeric_series = pd.to_numeric(df['angka'])
    overall_std = numeric_series.std()
    seed = int(numeric_series.mean())
    random.seed(seed)

    # Pengaturan Utama
    power_scale = min(data_size, 1000) / 1000.0
    settings['temperature'] = round(np.clip(0.5 + (overall_std / 5000) * 1.0 + random.uniform(-0.2, 0.2), 0.1, 2.0), 2)
    settings['power'] = round(1.0 + power_scale * 1.5 + random.uniform(-0.2, 0.2), 2)
    settings['min_conf'] = round(np.clip(0.01 - power_scale * 0.0095 + random.uniform(-0.0005, 0.0005), 0.0001, 0.01), 4)
    settings['voting_mode'] = random.choice(['product', 'average'])
    settings['mode_prediksi'] = random.choice(['confidence', 'ranked', 'hybrid'])

    # Window Size
    try:
        digits_df = df['angka'].str.zfill(4).apply(lambda x: [int(d) for d in x]).apply(pd.Series)
        digits_df.columns = [f"win_{label}" for label in DIGIT_LABELS]
        std_devs = digits_df.std()
        for label in DIGIT_LABELS:
            ws = 5 + (std_devs[f"win_{label}"] / 2.87) * 20 + random.uniform(-2, 2)
            settings[f"win_{label}"] = int(np.clip(ws, 3, 30))
    except Exception:
        for label in DIGIT_LABELS:
            settings[f"win_{label}"] = random.randint(5, 20)
            
    # Pengaturan Tambahan (yang sebelumnya tidak berubah)
    settings['cv_folds'] = int(np.clip(2 + power_scale * 8, 2, 10))
    settings['lstm_weight'] = round(random.uniform(0.5, 2.0), 2)
    settings['catboost_weight'] = round(random.uniform(0.5, 2.0), 2)
    settings['heatmap_weight'] = round(random.uniform(0.0, 1.0), 2)
    settings['min_conf_lstm'] = round(random.uniform(0.0, 1.0), 2)

    return settings

def initialize_state():
    defaults = {
        "temperature": 0.5, "voting_mode": "product", "power": 1.5, "min_conf": 0.0005,
        "mode_prediksi": "confidence", "win_ribuan": 7, "win_ratusan": 7, "win_puluhan": 7, "win_satuan": 7,
        "cv_folds": 3, "lstm_weight": 1.0, "catboost_weight": 1.0, "heatmap_weight": 0.5, "min_conf_lstm": 0.5
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def apply_dynamic_analysis():
    analisa_choice = st.session_state.get("analisa_choice")
    if analisa_choice and analisa_choice != "Tidak Ada":
        angka_list = st.session_state.get("angka_list", [])
        if len(angka_list) >= 10:
            df_for_analysis = pd.DataFrame({"angka": angka_list})
            settings = generate_dynamic_settings(df_for_analysis)
            if settings:
                for key, value in settings.items():
                    st.session_state[key] = value
                st.toast(f"💡 Setelan Analisa untuk '{analisa_choice}' telah diterapkan!")
        else:
            st.toast("⚠️ Tidak cukup data untuk Analisa (min. 10 angka).")

initialize_state()

# ======== Sidebar Pengaturan ========
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Hari", ["harian", "kemarin", "2hari", "3hari"])
    putaran = st.number_input("🔁 Putaran", 10, 1000, 100)
    metode = st.selectbox("🧠 Metode", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"])
    jumlah_uji = st.number_input("📊 Data Uji", 1, 200, 10)
    
    st.selectbox(
        "📈 Analisa", ["Tidak Ada"] + lokasi_list,
        key="analisa_choice",
        on_change=apply_dynamic_analysis,
        help="Pilih pasaran untuk menghasilkan & menerapkan setelan acak berdasarkan data saat ini."
    )
    
    temperature = st.slider("🌡️ Temperature", 0.1, 2.0, key="temperature", step=0.1)
    voting_mode = st.selectbox("⚖️ Kombinasi", ["product", "average"], key="voting_mode")
    power = st.slider("📈 Confidence Power", 0.5, 3.0, key="power", step=0.1)
    min_conf = st.slider("🔎 Min Confidence", 0.0001, 0.01, key="min_conf", format="%.4f")
    use_transformer = st.checkbox("🤖 Gunakan Transformer")
    model_type = "transformer" if use_transformer else "lstm"
    mode_prediksi = st.selectbox("🎯 Mode Prediksi", ["confidence", "ranked", "hybrid"], key="mode_prediksi")

    st.markdown("### 🪟 Window Size per Digit")
    window_per_digit = {}
    for label in DIGIT_LABELS:
        window_per_digit[label] = st.slider(f"{label.upper()}", 3, 30, key=f"win_{label}")
    # Pengaturan Tambahan dihapus dari sidebar untuk dipindahkan ke panel utama

# ======== Ambil Data API & Edit Manual ========
if "angka_list" not in st.session_state:
    st.session_state.angka_list = []

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Ambil Data dari API", use_container_width=True):
        try:
            with st.spinner("🔄 Mengambil data..."):
                url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
                headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
                data = requests.get(url, headers=headers).json()
                angka_api = [d["result"] for d in data["data"] if len(d["result"]) == 4 and d["result"].isdigit()]
                st.session_state.angka_list = angka_api
                st.success(f"{len(angka_api)} angka berhasil diambil.")
        except Exception as e:
            st.error(f"❌ Gagal ambil data: {e}")

with col2:
    st.caption("📌 Data angka akan digunakan untuk pelatihan dan prediksi")

with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    riwayat_input = "\n".join(st.session_state.angka_list)
    riwayat_input = st.text_area("📝 1 angka per baris:", value=riwayat_input, height=300)
    st.session_state.angka_list = [x.strip() for x in riwayat_input.splitlines() if x.strip().isdigit() and len(x.strip()) == 4]
    df = pd.DataFrame({"angka": st.session_state.get("angka_list", [])})

# ======== Tabs Utama ========
tab3_container, tab2, tab1 = st.tabs(["🔮 Scan Angka", "🪟 Scan Angka", "CatBoost"])

# ======== TAB 1 (Prediksi) ========
with tab1:
    # Kode di dalam Tab 1 tidak berubah
    # ...
    pass # Placeholder

# ======== TAB 2 (Scan Angka) ========
with tab2:
    min_ws = st.number_input("🔁 Min WS", 3, 10, 5)
    max_ws = st.number_input("🔁 Max WS", 4, 20, 11)
    
    # --- WIDGET PENGATURAN TAMBAHAN DIPINDAHKAN KE SINI ---
    st.number_input("Jumlah Fold", min_value=2, max_value=10, step=1, key="cv_folds")
    st.number_input("Seed", min_value=0, max_value=100, step=1, value=42)

    with st.container(border=True):
        st.subheader("⚖️ Bobot Voting")
        st.slider("LSTM Weight", 0.50, 2.00, key="lstm_weight", step=0.01)
        st.slider("CatBoost Weight", 0.50, 2.00, key="catboost_weight", step=0.01)
        st.slider("Heatmap Weight", 0.00, 1.00, key="heatmap_weight", step=0.01)
        st.slider("Min Confidence LSTM", 0.00, 1.00, key="min_conf_lstm", step=0.01)
    # --------------------------------------------------------
    
    use_cv = st.checkbox("Gunakan Cross Validation", value=False, key="use_cv_toggle")
    cv_folds_to_use = st.session_state.cv_folds if use_cv else None

    if st.button("🔍 Scan Angka (Normal)", use_container_width=True):
        with st.spinner("🔍 Mencari WS terbaik..."):
            # ... Logika scan Anda akan menggunakan nilai dari widget di atas ...
            st.success("Scan Selesai!")
    
# ======== TAB 3 (CatBoost) ========
with tab3_container:
    tab3(df, selected_lokasi)
