import streamlit as st
import pandas as pd
import requests
import os
import time

from markov_model import top6_markov, top6_markov_order2, top6_markov_hybrid
from ai_model import (
    top6_model,
    train_and_save_model,
    kombinasi_4d,
    find_best_window_size_with_model_true,
    build_lstm_model,
    build_transformer_model
)
from lokasi_list import lokasi_list

st.set_page_config(page_title="Prediksi AI", layout="wide")
st.title("Prediksi 4D - AI")

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

# Inisialisasi state jika belum ada
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

with tab_prediksi:
    if st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary"):
        pass # Isi dengan logika prediksi Anda

with tab_manajemen:
    st.subheader("Manajemen Model AI")
    if metode not in ["LSTM AI", "Ensemble AI + Markov"]:
        st.info("Pilih metode 'LSTM AI' atau 'Ensemble AI + Markov' di sidebar untuk mengelola model.")
    else:
        lokasi_id = selected_lokasi.lower().strip().replace(" ", "_")
        
        cols = st.columns(4)
        for i, label in enumerate(DIGIT_LABELS):
            with cols[i]:
                model_path = f"saved_models/{lokasi_id}_{label}_{model_type}.h5"
                st.markdown(f"##### {label.upper()}")
                if os.path.exists(model_path):
                    st.success("✅ Tersedia")
                    if st.button("Hapus", key=f"hapus_{label}", use_container_width=True):
                        os.remove(model_path)
                        st.rerun()
                else:
                    st.warning("⚠️ Belum ada")

        st.markdown("---")
        if st.button("📚 Latih & Simpan Semua Model AI", use_container_width=True):
            with st.spinner("🔄 Melatih semua model, ini mungkin memakan waktu..."):
                train_and_save_model(df, selected_lokasi, window_dict=window_per_digit, model_type=model_type)
            st.success("✅ Semua model berhasil dilatih dan disimpan.")
            st.rerun()

with tab_scan:
    st.subheader("Pencarian Window Size Optimal")
    st.info("Proses ini akan menjalankan scan untuk semua digit sekaligus dan menampilkan hasilnya. Proses ini bisa memakan waktu lama.")

    scan_cols = st.columns(4)
    with scan_cols[0]:
        min_ws = st.number_input("Min WS", 3, 10, 5, key="scan_min_ws")
    with scan_cols[1]:
        max_ws = st.number_input("Max WS", min_ws + 1, 30, 15, key="scan_max_ws")
    with scan_cols[2]:
        min_acc = st.slider("Min Akurasi", 0.0, 1.0, 0.05, key="scan_min_acc")
    with scan_cols[3]:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.05, key="scan_min_conf")
    
    if st.button("🔎 Jalankan Scan untuk Semua Digit", use_container_width=True, type="primary"):
        if len(df) < max_ws + 5:
            st.error(f"Data tidak cukup. Dibutuhkan setidaknya {max_ws + 5} baris data.")
        else:
            st.subheader("Hasil Scan Window Size")
            st.warning("Harap tunggu, proses untuk semua digit sedang berjalan...")
            
            # Buat placeholder untuk hasil
            results_placeholder = st.empty()
            hasil_scan_semua = {}

            for label in DIGIT_LABELS:
                best_ws, _ = find_best_window_size_with_model_true(
                    df, label, selected_lokasi, model_type=model_type,
                    min_ws=min_ws, max_ws=max_ws, temperature=temperature,
                    top_n=jumlah_digit, min_acc=min_acc, min_conf=min_conf
                )
                
                if best_ws is not None:
                    hasil_scan_semua[label.upper()] = best_ws
                else:
                    hasil_scan_semua[label.upper()] = "Tidak Ditemukan"
                
                # Tampilkan hasil sementara
                results_placeholder.json(hasil_scan_semua)
            
            # Tampilkan hasil akhir
            results_placeholder.empty() # Hapus tabel sementara
            st.success("✅ Scan Selesai!")
            st.markdown("Berikut adalah hasil WS terbaik yang ditemukan:")
            
            # Tampilkan hasil akhir dalam format yang lebih baik
            final_cols = st.columns(4)
            for i, (label, ws) in enumerate(hasil_scan_semua.items()):
                final_cols[i].metric(label, f"WS: {ws}")

            st.info("Silakan **atur slider di sidebar secara manual** sesuai hasil di atas, lalu latih ulang model di tab 'Manajemen Model'.")
