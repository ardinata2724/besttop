import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

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

st.set_page_config(page_title="Prediksi AI", layout="wide")
st.title("Prediksi 4D - AI")

# --- PERBAIKAN STATE MANAGEMENT DITAMBAHKAN DI SINI ---
# Langkah 3: Terapkan hasil scan dari session_state sementara SEBELUM widget dirender.
if 'scan_results_to_apply' in st.session_state:
    for key, value in st.session_state.scan_results_to_apply.items():
        st.session_state[key] = value
    # Hapus state sementara setelah diterapkan
    del st.session_state.scan_results_to_apply

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

# Inisialisasi state jika belum ada
for label in DIGIT_LABELS:
    key = f"win_{label}"
    if key not in st.session_state:
        st.session_state[key] = 7

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Hari", ["harian", "kemarin", "2hari", "3hari"])
    putaran = st.number_input("🔁 Putaran", 10, 1000, 100)
    
    st.markdown("### 🎯 Opsi Prediksi")
    jumlah_digit = st.slider("🔢 Jumlah Digit Prediksi", 1, 9, 6)
    
    metode = st.selectbox("🧠 Metode", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"])
    use_transformer = st.checkbox("🤖 Gunakan Transformer")
    model_type = "transformer" if use_transformer else "lstm"

    st.markdown("### ⚙️ Parameter Lanjutan")
    temperature = st.slider("🌡️ Temperature", 0.1, 2.0, 0.5, step=0.1)
    mode_prediksi = st.selectbox("🎯 Mode Prediksi AI", ["confidence", "ranked", "hybrid"])
    
    with st.expander("Kombinasi 4D"):
        voting_mode = st.selectbox("⚖️ Metode Kombinasi", ["product", "average"])
        power = st.slider("📈 Confidence Power", 0.5, 3.0, 1.5, 0.1)
        min_conf_kombinasi = st.slider("🔎 Min Confidence Kombinasi", 0.0001, 0.01, 0.0005, 0.0001, format="%.4f")

    st.markdown("### 🪟 Window Size per Digit")
    window_per_digit = {}
    for label in DIGIT_LABELS:
        # Slider akan mengambil nilai dari st.session_state yang sudah diperbarui di atas
        window_per_digit[label] = st.slider(
            f"{label.upper()}", 3, 30, st.session_state[f"win_{label}"], key=f"win_{label}"
        )

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
    st.caption("📌 Data angka akan digunakan untuk pelatihan dan prediksi.")

with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    riwayat_input = "\n".join(st.session_state.angka_list)
    riwayat_input = st.text_area("📝 1 angka per baris:", value=riwayat_input, height=300)
    st.session_state.angka_list = [x.strip() for x in riwayat_input.splitlines() if x.strip().isdigit() and len(x.strip()) == 4]
    df = pd.DataFrame({"angka": st.session_state.angka_list})

# ======== Tabs Utama ========
tab_prediksi, tab_scan, tab_manajemen = st.tabs(["🔮 Prediksi & Hasil", "🪟 Scan Window Size", "⚙️ Manajemen Model"])

# ... (Tab Prediksi dan Manajemen Model tidak berubah, kodenya tetap sama) ...
with tab_prediksi:
    # ... (Isi tab ini sama seperti sebelumnya) ...
    pass

with tab_manajemen:
    # ... (Isi tab ini sama seperti sebelumnya) ...
    pass

# --- BLOK SCAN WINDOW SIZE DIPERBAIKI SECARA TOTAL ---
with tab_scan:
    st.subheader("Pencarian Window Size Optimal dengan Model Training")
    st.info("Proses ini akan melatih model sementara untuk setiap window size guna mencari akurasi terbaik. Proses ini bisa memakan waktu.")

    scan_cols = st.columns(4)
    with scan_cols[0]:
        min_ws = st.number_input("Min WS", 3, 10, 5, key="scan_min_ws")
    with scan_cols[1]:
        max_ws = st.number_input("Max WS", min_ws + 1, 30, 15, key="scan_max_ws")
    with scan_cols[2]:
        min_acc = st.slider("Min Akurasi", 0.0, 1.0, 0.05, key="scan_min_acc") # Default diturunkan
    with scan_cols[3]:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.05, key="scan_min_conf") # Default diturunkan
    
    if st.button("🔎 Scan Semua Digit", use_container_width=True, type="primary"):
        if len(df) < max_ws + 5:
            st.error(f"Data tidak cukup. Dibutuhkan setidaknya {max_ws + 5} baris data.")
        else:
            # Langkah 1: Jalankan scan dan simpan hasilnya di dictionary LOKAL
            hasil_scan_lokal = {}
            for label in DIGIT_LABELS:
                with st.expander(f"Log Scan untuk {label.upper()}", expanded=True):
                    with st.spinner(f"Mencari WS terbaik untuk {label.upper()}..."):
                        best_ws, top_n_digits = find_best_window_size_with_model_true(
                            df, label, selected_lokasi, model_type=model_type,
                            min_ws=min_ws, max_ws=max_ws, temperature=temperature,
                            top_n=jumlah_digit, min_acc=min_acc, min_conf=min_conf
                        )
                        if best_ws is not None:
                            st.success(f"WS terbaik untuk {label.upper()}: {best_ws}.")
                            hasil_scan_lokal[f"win_{label}"] = best_ws
                        else:
                            st.warning(f"Tidak ditemukan WS yang cocok untuk {label.upper()} dengan kriteria yang diberikan.")
            
            # Langkah 2: Jika ada hasil, simpan ke state SEMENTARA dan paksa RERUN
            if hasil_scan_lokal:
                st.session_state.scan_results_to_apply = hasil_scan_lokal
                st.rerun()
