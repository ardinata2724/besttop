import streamlit as st
import pandas as pd
import requests
import os
import time
import random

from markov_model import top6_markov, top6_markov_order2, top6_markov_hybrid
from ai_model import (
    top6_model,
    train_and_save_model,
    kombinasi_4d,
    find_best_window_size_with_model_true,
    build_lstm_model,
    build_transformer_model,
    top6_ensemble
)
from lokasi_list import lokasi_list

st.set_page_config(page_title="Prediksi AI", layout="wide")
st.title("Prediksi 4D - AI")

# Inisialisasi state untuk menyimpan hasil scan yang persisten
if 'scan_outputs' not in st.session_state:
    st.session_state.scan_outputs = {}

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

for label in DIGIT_LABELS:
    if f"win_{label}" not in st.session_state:
        st.session_state[f"win_{label}"] = 7

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

# ... Kode untuk Ambil Data & Edit Manual tidak berubah ...
if "angka_list" not in st.session_state:
    st.session_state.angka_list = []
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Ambil Data dari API", use_container_width=True):
        st.session_state.angka_list = [] # Reset data
        # ... sisa logika ambil data
with col2:
    st.caption("📌 Data angka akan digunakan untuk pelatihan dan prediksi.")
with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    # ... sisa logika edit manual
    pass
df = pd.DataFrame({"angka": st.session_state.get("angka_list", [])})


# ======== Tabs Utama ========
tab_prediksi, tab_scan, tab_manajemen = st.tabs(["🔮 Prediksi & Hasil", "🪟 Scan Window Size", "⚙️ Manajemen Model"])

with tab_prediksi:
    # ... Kode tab prediksi tidak berubah ...
    pass

with tab_manajemen:
    # ... Kode tab manajemen tidak berubah ...
    pass

with tab_scan:
    st.subheader("Pencarian Window Size Optimal")
    st.info("Jalankan scan untuk setiap digit. Hasil scan akan tersimpan dan ditampilkan di bawah secara berurutan.")

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

    btn_cols = st.columns(4)
    for i, label in enumerate(DIGIT_LABELS):
        if btn_cols[i].button(f"🔎 Scan {label.upper()}", use_container_width=True):
            if len(df) < max_ws + 5:
                st.error(f"Data tidak cukup. Dibutuhkan setidaknya {max_ws + 5} baris data.")
            else:
                with st.spinner(f"Mencari WS terbaik untuk {label.upper()}..."):
                    best_ws, result_table = find_best_window_size_with_model_true(
                        df, label, selected_lokasi, model_type=model_type,
                        min_ws=min_ws, max_ws=max_ws, temperature=temperature,
                        top_n=jumlah_digit, min_acc=min_acc, min_conf=min_conf
                    )
                    # Simpan hasilnya ke session_state
                    st.session_state.scan_outputs[label] = {
                        "best_ws": best_ws,
                        "table": result_table
                    }
                    # Update slider di sidebar secara manual jika ada hasil
                    if best_ws is not None:
                        st.session_state[f"win_{label}"] = best_ws

    # Tombol untuk mereset/menghapus semua hasil scan
    if st.button("❌ Hapus Semua Hasil Scan"):
        st.session_state.scan_outputs = {}
        st.rerun()

    st.divider()

    # --- BAGIAN BARU: Tampilkan semua hasil yang tersimpan ---
    if not st.session_state.scan_outputs:
        st.info("Belum ada hasil scan. Silakan klik tombol scan di atas.")
    else:
        st.subheader("📜 Riwayat Hasil Scan")
        # Urutkan berdasarkan urutan digit
        sorted_labels = [label for label in DIGIT_LABELS if label in st.session_state.scan_outputs]
        for label in sorted_labels:
            output = st.session_state.scan_outputs[label]
            with st.expander(f"Hasil untuk {label.upper()}", expanded=True):
                if output["table"] is not None:
                    st.dataframe(output["table"])
                    if output["best_ws"] is not None:
                        st.success(f"✅ {label.upper()} - WS terbaik yang ditemukan: {output['best_ws']}")
                    else:
                        st.error(f"❌ {label.upper()} - Tidak ada Window Size yang memenuhi kriteria.")
                else:
                    st.error(f"❌ {label.upper()} - Tidak ada Window Size yang memenuhi kriteria.")
