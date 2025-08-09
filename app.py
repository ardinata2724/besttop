import streamlit as st
import pandas as pd
import requests
import os
import time
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter
from itertools import product
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dropout, Dense,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from datetime import datetime

# ==============================================================================
# BAGIAN 1: DEFINISI FUNGSI-FUNGSI INTI
# ==============================================================================

@st.cache_data(ttl=600)
def fetch_live_results(pasaran_list):
    results = []
    headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
    for pasaran in pasaran_list:
        try:
            url = f"https://wysiwygscan.com/api?pasaran={pasaran.lower()}&hari=harian&putaran=1&format=json&urut=desc"
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("data"):
                latest_result = data["data"][0]
                tanggal = datetime.strptime(latest_result['tanggal'], '%Y-%m-%d').strftime('%d-%m-%Y')
                results.append({
                    "Pasaran": pasaran.capitalize(),
                    "Tanggal": tanggal,
                    "Hasil": latest_result["result"]
                })
            else:
                results.append({"Pasaran": pasaran.capitalize(), "Tanggal": "-", "Hasil": "-"})
        except Exception:
            results.append({"Pasaran": pasaran.capitalize(), "Tanggal": "Error", "Hasil": "Gagal"})
    return results

# ... (Semua fungsi inti lainnya tetap ada di sini, tidak perlu diubah)
DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

# ==============================================================================
# BAGIAN 2: APLIKASI STREAMLIT UTAMA
# ==============================================================================

st.set_page_config(page_title="Prediksi AI", layout="wide")
st.title("Prediksi 4D - AI")

try:
    from lokasi_list import lokasi_list
except ImportError:
    lokasi_list = ["Taipei", "NCD", "Cambodia", "Bullseye", "Sydney", "Sdy Lotto", "China", "Japan", "Singapore", "Mongolia", "Pcso", "Taiwan", "Osaka", "Nusantara", "Hongkong", "Hkg Lotto"]

# Inisialisasi state
for label in DIGIT_LABELS:
    if f"win_{label}" not in st.session_state:
        st.session_state[f"win_{label}"] = 7
if "angka_list" not in st.session_state:
    st.session_state.angka_list = []

# --- UI Sidebar ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Hari", ["harian", "kemarin", "2hari", "3hari"])
    putaran = st.number_input("🔁 Putaran", 10, 1000, 100)
    st.markdown("### 🎯 Opsi Prediksi")
    jumlah_digit = st.slider("🔢 Jumlah Digit Prediksi", 1, 9, 6)
    metode = st.selectbox("🧠 Metode", ["Markov", "LSTM AI", "Ensemble AI + Markov"])
    use_transformer = st.checkbox("🤖 Gunakan Transformer", value=True)
    model_type = "transformer" if use_transformer else "lstm"
    st.markdown("### 🪟 Window Size per Digit")
    window_per_digit = {}
    for label in DIGIT_LABELS:
        window_per_digit[label] = st.slider(f"{label.upper()}", 3, 30, st.session_state[f"win_{label}"], key=f"win_{label}")

    st.markdown("---") # Pemisah

    # --- TAMPILKAN LIVE RESULT DI SINI (DI BAWAH) ---
    st.subheader("🔴 Live Result")
    live_results_data = fetch_live_results(lokasi_list)
    if live_results_data:
        df_live = pd.DataFrame(live_results_data)
        st.table(df_live)
    else:
        st.warning("Gagal memuat live result.")

    # --- Skrip untuk refresh otomatis ---
    refresh_interval = 600  # 10 menit
    js_code = f"""
    <script>
    setTimeout(function() {{
        window.parent.location.reload();
    }}, {refresh_interval * 1000});
    </script>
    """
    st.html(js_code, height=0, width=0)

# --- Sisa UI dan Logika Aplikasi (tidak ada perubahan) ---
# ... (Kode untuk Ambil Data, Edit Manual, dan semua Tab) ...
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Ambil Data dari API", use_container_width=True):
        pass # Isi dengan logika Anda
with col2:
    st.caption("Data angka akan digunakan untuk pelatihan dan prediksi.")
with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    pass # Isi dengan logika Anda
df = pd.DataFrame({"angka": st.session_state.get("angka_list", [])})

tab_prediksi, tab_scan, tab_manajemen = st.tabs(["🔮 Prediksi & Hasil", "🪟 Scan Window Size", "⚙️ Manajemen Model"])
with tab_prediksi:
    pass
with tab_manajemen:
    pass
with tab_scan:
    pass
