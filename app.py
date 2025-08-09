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
DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

# --- Fungsi dari Markov Model ---
def _ensure_unique_top_n(top_list, n=6):
    unique_list = list(dict.fromkeys(top_list))[:n]
    if len(unique_list) >= n: return unique_list
    all_digits = list(range(10)); random.shuffle(all_digits)
    unique_set = set(unique_list)
    for digit in all_digits:
        if len(unique_set) >= n: break
        if digit not in unique_set: unique_set.add(digit)
    return list(unique_set)

def top6_markov(df, top_n=6):
    if df.empty or len(df) < 10: return [], None
    data = df["angka"].astype(str).tolist()
    matrix = [defaultdict(lambda: defaultdict(int)) for _ in range(3)]
    for number in data:
        digits = f"{int(number):04d}"
        for i in range(3): matrix[i][digits[i]][digits[i+1]] += 1
    freq_ribuan = Counter([int(x[0]) for x in data])
    hasil = [[k for k, _ in freq_ribuan.most_common(top_n)]]
    for i in range(3):
        kandidat = [int(k) for prev in matrix[i] for k in matrix[i][prev].keys()]
        top = [k for k, _ in Counter(kandidat).most_common()]
        hasil.append(top)
    return [_ensure_unique_top_n(h, n=top_n) for h in hasil], None

# --- Fungsi dari AI Model (disederhanakan untuk contoh) ---
def top_n_model(df, lokasi, window_dict, model_type, top_n=6):
    # Placeholder: Mengembalikan hasil acak untuk demonstrasi
    st.warning("Fungsi model AI belum diimplementasikan sepenuhnya di versi ini.")
    return [random.sample(range(10), top_n) for _ in range(4)], None

# ==============================================================================
# BAGIAN 2: APLIKASI STREAMLIT UTAMA
# ==============================================================================

st.set_page_config(page_title="Prediksi AI", layout="wide")
st.title("Prediksi 4D - AI")

# Inisialisasi
if "angka_list" not in st.session_state: st.session_state.angka_list = []
try:
    from lokasi_list import lokasi_list
except ImportError:
    lokasi_list = ["BULLSEYE", "HONGKONG", "SYDNEY", "SINGAPORE"]

# --- UI Sidebar ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Hari", ["harian", "kemarin", "2hari", "3hari"])
    putaran = st.number_input("🔁 Putaran", 10, 1000, 100)
    st.markdown("---")
    st.markdown("### 🎯 Opsi Prediksi")
    jumlah_digit = st.slider("🔢 Jumlah Digit Prediksi", 1, 9, 7)
    metode = st.selectbox("🧠 Metode", ["Markov", "LSTM AI"])
    st.markdown("---")
    st.markdown("### 🪟 Window Size per Digit")
    window_per_digit = {}
    for label in DIGIT_LABELS:
        if f"win_{label}" not in st.session_state:
            st.session_state[f"win_{label}"] = 7
        window_per_digit[label] = st.slider(f"{label.upper()}", 3, 30, st.session_state[f"win_{label}"], key=f"win_{label}")

# --- UI Data Loader ---
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Ambil Data dari API", use_container_width=True):
        try:
            with st.spinner("🔄 Mengambil data..."):
                url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
                headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                if data.get("data"):
                    angka_api = [d["result"] for d in data["data"] if len(str(d.get("result", ""))) == 4 and str(d.get("result", "")).isdigit()]
                    st.session_state.angka_list = angka_api
                    st.success(f"{len(angka_api)} angka berhasil diambil.")
                else:
                    st.error("API tidak mengembalikan data yang valid.")
        except Exception as e:
            st.error(f"Gagal mengambil data dari API: {e}")

with col2: st.caption("Data angka akan digunakan untuk pelatihan dan prediksi.")

with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    riwayat_input = "\n".join(st.session_state.get("angka_list", []))
    riwayat_text = st.text_area("1 angka per baris:", riwayat_input, height=250, key="manual_input")
    if riwayat_text != riwayat_input:
        st.session_state.angka_list = [x.strip() for x in riwayat_text.splitlines() if x.strip().isdigit() and len(x.strip()) == 4]
        st.rerun()
df = pd.DataFrame({"angka": st.session_state.get("angka_list", [])})

# ======== Tabs Utama ========
tab_prediksi, tab_scan, tab_manajemen = st.tabs(["🔮 Prediksi & Hasil", "🪟 Scan Window Size", "⚙️ Manajemen Model"])

with tab_prediksi:
    if st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary"):
        max_ws = max(window_per_digit.values())
        if len(df) < max_ws + 1:
            st.warning(f"❌ Data tidak cukup. Butuh minimal {max_ws + 1} baris.")
        else:
            result, _ = None, None
            with st.spinner("⏳ Memproses prediksi..."):
                if metode == "Markov":
                    result, _ = top6_markov(df, top_n=jumlah_digit)
                elif metode == "LSTM AI":
                    result, _ = top_n_model(df, selected_lokasi, window_per_digit, "lstm", jumlah_digit)

            if result and all(result):
                st.subheader(f"🎯 Hasil Prediksi Top {jumlah_digit}")
                for i, label in enumerate(DIGIT_LABELS):
                    st.markdown(f"**{label.upper()}:** {', '.join(map(str, result[i]))}")
                
                st.divider()
                
                # --- BLOK BARU UNTUK KOMBINASI PENUH 4D (BBFS) ---
                all_combinations = list(product(*result))
                kombinasi_4d_list = ["".join(map(str, combo)) for combo in all_combinations]
                total_kombinasi = len(kombinasi_4d_list)
                
                st.subheader(f"🔢 Semua Kombinasi 4D ({total_kombinasi} Line)")
                output_string = " * ".join(kombinasi_4d_list)
                st.text_area(f"Total {total_kombinasi} Kombinasi Penuh (dipisah dengan '*')", output_string, height=300)

with tab_scan:
    st.warning("Fitur 'Scan Window Size' sedang dalam perbaikan dan akan segera diaktifkan kembali.")

with tab_manajemen:
    st.warning("Fitur 'Manajemen Model' sedang dalam perbaikan dan akan segera diaktifkan kembali.")
