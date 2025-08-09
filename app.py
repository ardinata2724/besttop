import streamlit as st
import pandas as pd
import requests
import os
import time
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter
from itertools import product  # <-- Penting untuk membuat semua kombinasi
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
# BAGIAN 1: SEMUA FUNGSI-FUNGSI INTI
# (Tidak ada perubahan di bagian ini)
# ==============================================================================
DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

# ... (Semua fungsi inti dari versi sebelumnya tetap di sini) ...
def _ensure_unique_top_n(top_list, n=6):
    # ...
    pass
def top6_markov(df, top_n=6):
    # ...
    pass
# ... dan semua fungsi AI lainnya ...

# ==============================================================================
# BAGIAN 2: APLIKASI STREAMLIT UTAMA
# ==============================================================================

st.set_page_config(page_title="Prediksi AI", layout="wide")
st.title("Prediksi 4D - AI")

try: from lokasi_list import lokasi_list
except ImportError: lokasi_list = ["HONGKONG", "BULLSEYE", "SYDNEY", "SINGAPORE"]

if 'scan_outputs' not in st.session_state: st.session_state.scan_outputs = {}
for label in DIGIT_LABELS:
    if f"win_{label}" not in st.session_state: st.session_state[f"win_{label}"] = 7
if "angka_list" not in st.session_state: st.session_state.angka_list = []

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Hari", ["harian", "kemarin", "2hari", "3hari"])
    putaran = st.number_input("🔁 Putaran", 10, 1000, 100)
    st.markdown("---")
    st.markdown("### 🎯 Opsi Prediksi")
    jumlah_digit = st.slider("🔢 Jumlah Digit Prediksi", 1, 9, 7)
    metode = st.selectbox("🧠 Metode", ["Markov", "LSTM AI"])
    use_transformer = st.checkbox("🤖 Gunakan Transformer", value=True)
    model_type = "transformer" if use_transformer else "lstm"
    st.markdown("---")
    st.markdown("### 🪟 Window Size per Digit")
    window_per_digit = {}
    for label in DIGIT_LABELS:
        window_per_digit[label] = st.slider(f"{label.upper()}", 3, 30, st.session_state[f"win_{label}"], key=f"win_{label}")

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Ambil Data dari API", use_container_width=True):
        # ... logika ambil data
        pass
with col2:
    st.caption("Data angka akan digunakan untuk pelatihan dan prediksi.")
with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    # ... logika edit manual
    pass
df = pd.DataFrame({"angka": st.session_state.get("angka_list", [])})

tab_prediksi, tab_scan, tab_manajemen = st.tabs(["🔮 Prediksi & Hasil", "🪟 Scan Window Size", "⚙️ Manajemen Model"])

with tab_prediksi:
    if st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary"):
        max_ws_needed = max(window_per_digit.values())
        if len(df) < max_ws_needed + 1:
            st.warning(f"❌ Data tidak cukup. Butuh minimal {max_ws_needed + 1} baris.")
        else:
            result, _ = None, None
            with st.spinner("⏳ Memproses prediksi..."):
                if metode == "Markov": result, _ = top6_markov(df, top_n=jumlah_digit)
            
            if result:
                st.subheader(f"🎯 Hasil Prediksi Top {jumlah_digit}")
                for i, label in enumerate(DIGIT_LABELS):
                    st.markdown(f"**{label.upper()}:** {', '.join(map(str, result[i]))}")

                # --- BLOK BARU UNTUK KOMBINASI PENUH 4D (BBFS) ---
                st.divider()
                
                if all(result) and len(result) == 4:
                    # Menghasilkan semua kemungkinan kombinasi (Cartesian product)
                    all_combinations = list(product(*result))
                    
                    # Mengubah setiap tuple kombinasi menjadi string 4D
                    kombinasi_4d_list = ["".join(map(str, combo)) for combo in all_combinations]
                    
                    # Menghitung jumlah total kombinasi yang dihasilkan
                    total_kombinasi = len(kombinasi_4d_list)
                    
                    st.subheader(f"🔢 Semua Kombinasi 4D ({total_kombinasi} Line)")

                    # Menggabungkan semua angka dengan pemisah bintang
                    output_string = " * ".join(kombinasi_4d_list)
                    
                    st.text_area(f"{total_kombinasi} Kombinasi Penuh (dipisah dengan '*')", output_string, height=300)
                else:
                    st.warning("Tidak bisa menghasilkan kombinasi karena hasil prediksi tidak lengkap.")
                # --- AKHIR BLOK BARU ---

with tab_manajemen:
    # ... (Isi tab ini tidak berubah)
    pass

with tab_scan:
    # ... (Isi tab ini tidak berubah)
    pass
