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
import locale

# ==============================================================================
# BAGIAN 1: DEFINISI SEMUA FUNGSI-FUNGSI INTI
# (Tidak ada perubahan di bagian ini)
# ==============================================================================
DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

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

# ... (Semua fungsi AI lainnya tetap ada di sini)

# ==============================================================================
# BAGIAN 2: APLIKASI STREAMLIT UTAMA
# ==============================================================================

st.set_page_config(page_title="Prediksi 4D", layout="wide")

# --- PERUBAHAN DI SINI: MENAMBAHKAN INFORMASI WAKTU REALTIME ---
# Atur locale ke bahasa Indonesia
try:
    locale.setlocale(locale.LC_TIME, 'id_ID.UTF-8')
except locale.Error:
    locale.setlocale(locale.LC_TIME, 'Indonesian_Indonesia.1252')

# Dapatkan waktu saat ini
now = datetime.now()
hari = now.strftime('%A')
tanggal = now.strftime('%d %B %Y')

# Tampilkan informasi waktu dan lokasi
st.markdown(f"**Brebes, {hari}, {tanggal}**")

# --- PERUBAHAN JUDUL & COPYRIGHT ---
st.title("Prediksi 4D")
st.caption("Create by: ANDI")


# --- Sisa Kode Aplikasi ---
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
    riwayat_input = "\n".join(st.session_state.get("angka_list", []))
    riwayat_text = st.text_area("1 angka per baris:", riwayat_input, height=250)
    if riwayat_text != riwayat_input:
        st.session_state.angka_list = [x.strip() for x in riwayat_text.splitlines() if x.strip().isdigit() and len(x.strip()) == 4]
        st.rerun()
df = pd.DataFrame({"angka": st.session_state.get("angka_list", [])})

tab_prediksi, tab_scan, tab_manajemen = st.tabs(["🔮 Prediksi & Hasil", "🪟 Scan Window Size", "⚙️ Manajemen Model"])

with tab_prediksi:
    if st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary"):
        # ... (Logika prediksi tidak berubah)
        pass

with tab_manajemen:
    st.subheader("Manajemen Model AI")
    # ... (Logika manajemen tidak berubah)
    pass

with tab_scan:
    st.subheader("Pencarian Window Size (WS) Optimal per Digit")
    st.info("Klik tombol scan untuk setiap digit. Hasilnya akan muncul dan tetap ada di bawah. Setelah menemukan WS terbaik, **atur slider di sidebar secara manual**.")
    # ... (Logika scan tidak berubah)
    pass
