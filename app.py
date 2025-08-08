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

# --- MANAJEMEN STATE UNTUK SCAN BERTAHAP ---
if 'scan_status' not in st.session_state:
    st.session_state.scan_status = 'idle'  # idle, scanning, finished
if 'scan_current_digit_index' not in st.session_state:
    st.session_state.scan_current_digit_index = 0
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = {}

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

# Placeholder untuk tab yang tidak ditampilkan
with tab_prediksi:
    st.subheader("Prediksi & Hasil")
    # ... (Kode tab prediksi tetap sama)

with tab_manajemen:
    st.subheader("Manajemen Model AI")
    # ... (Kode tab manajemen tetap sama)

# --- BLOK SCAN WINDOW SIZE DIPERBAIKI DENGAN LOGIKA CALLBACK ---
with tab_scan:
    st.subheader("Pencarian Window Size Optimal (Bertahap & Otomatis)")
    st.info("Proses ini akan mencari WS optimal untuk setiap digit secara berurutan. Klik 'Mulai Scan' untuk memulai, aplikasi akan berjalan otomatis hingga selesai.")

    scan_cols = st.columns(4)
    with scan_cols[0]:
        min_ws = st.number_input("Min WS", 3, 10, 5, key="scan_min_ws")
    with scan_cols[1]:
        max_ws = st.number_input("Max WS", min_ws + 1, 30, 15, key="scan_max_ws")
    with scan_cols[2]:
        min_acc = st.slider("Min Akurasi", 0.0, 1.0, 0.05, key="scan_min_acc")
    with scan_cols[3]:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.05, key="scan_min_conf")

    st.divider()

    # --- Fungsi Callback untuk mengubah state ---
    def start_scan():
        st.session_state.scan_status = 'scanning'
        st.session_state.scan_current_digit_index = 0
        st.session_state.scan_results = {}

    def reset_scan():
        st.session_state.scan_status = 'idle'
        st.session_state.scan_current_digit_index = 0
        st.session_state.scan_results = {}

    btn_cols = st.columns(2)
    with btn_cols[0]:
        st.button(
            "🚀 Mulai Scan Bertahap",
            on_click=start_scan,
            use_container_width=True,
            type="primary",
            disabled=(st.session_state.scan_status == 'scanning')
        )
            
    with btn_cols[1]:
        st.button("🔄 Reset Scan", on_click=reset_scan, use_container_width=True)

    # --- Menampilkan hasil yang sudah terkumpul ---
    if st.session_state.scan_results:
        st.subheader("Hasil Scan Sementara")
        res_cols = st.columns(4)
        for i, label in enumerate(DIGIT_LABELS):
            if label in st.session_state.scan_results:
                ws = st.session_state.scan_results[label]
                res_cols[i].metric(label=label.upper(), value=f"WS: {ws}")

    # --- Logika utama untuk menjalankan scan, hanya jika statusnya 'scanning' ---
    if st.session_state.scan_status == 'scanning':
        idx = st.session_state.scan_current_digit_index
        
        if idx < len(DIGIT_LABELS):
            label = DIGIT_LABELS[idx]
            
            # Tampilkan pesan proses
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                st.info(f"⚙️ Sedang memproses {label.upper()} ({idx + 1}/{len(DIGIT_LABELS)})... Ini mungkin perlu waktu beberapa menit.")
                st.progress((idx) / len(DIGIT_LABELS))

            best_ws, _ = find_best_window_size_with_model_true(
                df, label, selected_lokasi, model_type=model_type,
                min_ws=min_ws, max_ws=max_ws, temperature=temperature,
                top_n=jumlah_digit, min_acc=min_acc, min_conf=min_conf
            )
            
            if best_ws is not None:
                st.session_state[f"win_{label}"] = best_ws
                st.session_state.scan_results[label] = best_ws
            else:
                st.session_state.scan_results[label] = "Gagal"
            
            # Pindah ke digit selanjutnya
            st.session_state.scan_current_digit_index += 1
            # Hapus placeholder dan rerun untuk proses selanjutnya
            progress_placeholder.empty()
            st.rerun()
        else:
            # Jika semua sudah selesai
            st.session_state.scan_status = 'finished'
            st.rerun()
            
    elif st.session_state.scan_status == 'finished':
        st.success("🎉 Semua digit telah selesai di-scan!")
        st.info("Pengaturan Window Size di sidebar telah diperbarui. Anda bisa melatih ulang model di tab 'Manajemen Model' sekarang.")
        st.balloons()
        # Reset status setelah selesai
        st.session_state.scan_status = 'idle'
        st.session_state.scan_current_digit_index = 0
