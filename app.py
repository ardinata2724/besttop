import streamlit as st
import pandas as pd
import requests
import os
import time
import random # Ditambahkan untuk fitur acak

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
        max_ws_needed = max(list(window_per_digit.values()))
        if len(df) < max_ws_needed + 1:
            st.warning(f"❌ Data tidak cukup. Metode AI membutuhkan setidaknya {max_ws_needed + 1} baris data untuk window size {max_ws_needed}.")
        else:
            with st.spinner("⏳ Memproses prediksi..."):
                result, probs = None, None
                if metode == "Markov":
                    result, _ = top6_markov(df, top_n=jumlah_digit)
                elif metode == "Markov Order-2":
                    result = top6_markov_order2(df, top_n=jumlah_digit)
                elif metode == "Markov Gabungan":
                    result = top6_markov_hybrid(df, top_n=jumlah_digit)
                
                elif metode == "LSTM AI":
                    pred_data = top6_model(
                        df, lokasi=selected_lokasi, model_type=model_type,  
                        return_probs=True, temperature=temperature,  
                        mode_prediksi=mode_prediksi, window_dict=window_per_digit,
                        top_n=jumlah_digit
                    )
                    if pred_data:
                        result, probs = pred_data
                    else:
                        st.error("Gagal memuat model AI. Pastikan model sudah dilatih di tab 'Manajemen Model'.")
                
                elif metode == "Ensemble AI + Markov":
                    result = top6_ensemble(
                        df, lokasi=selected_lokasi, model_type=model_type,
                        window_dict=window_per_digit, temperature=temperature,
                        mode_prediksi=mode_prediksi, top_n=jumlah_digit
                    )
                    probs = None 
                    if result is None:
                        st.error("Gagal melakukan prediksi ensemble. Pastikan model AI sudah dilatih.")
            
            if result:
                st.subheader(f"🎯 Hasil Prediksi Top {jumlah_digit}")
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

                # --- BLOK BARU UNTUK ACAK 4D ---
                st.divider()
                st.subheader("🎲 Acak 4D dari Hasil Prediksi")
                
                # Pastikan hasil prediksi valid sebelum diacak
                if all(result):
                    try:
                        acak_4d_list = []
                        for _ in range(1000):
                            d1 = random.choice(result[0])
                            d2 = random.choice(result[1])
                            d3 = random.choice(result[2])
                            d4 = random.choice(result[3])
                            acak_4d_list.append(f"{d1}{d2}{d3}{d4}")
                        
                        # Gabungkan semua angka dengan pemisah bintang
                        output_string = " * ".join(acak_4d_list)
                        
                        st.text_area("1000 Kombinasi Acak (dipisah dengan '*')", output_string, height=300)
                    except IndexError:
                        st.error("Gagal menghasilkan angka acak, pastikan setiap posisi digit memiliki hasil prediksi.")
                else:
                    st.warning("Tidak bisa menghasilkan angka acak karena salah satu hasil prediksi kosong.")
                # --- AKHIR BLOK BARU ---

                if probs:
                    st.subheader("📊 Confidence Bar")
                    for i, label in enumerate(DIGIT_LABELS):
                        if result[i] and probs[i]:
                            st.markdown(f"**{label.upper()}**")
                            dconf = pd.DataFrame({
                                "Digit": [str(d) for d in result[i]],
                                "Confidence": probs[i]
                            }).sort_values("Confidence", ascending=True)
                            st.bar_chart(dconf.set_index("Digit"))

                if metode in ["LSTM AI"]:
                    with st.spinner("🔢 Menghitung kombinasi 4D..."):
                        top_komb = kombinasi_4d(
                            df, lokasi=selected_lokasi, model_type=model_type,
                            top_n=20, min_conf=min_conf_kombinasi, power=power,
                            mode=voting_mode, window_dict=window_per_digit,
                            mode_prediksi=mode_prediksi,
                            pred_n=jumlah_digit
                        )
                        st.subheader("💡 Kombinasi 4D Populer (Berdasarkan Confidence)")
                        if top_komb:
                            for komb, score in top_komb:
                                st.markdown(f"`{komb}` - Skor Keyakinan: `{score:.6f}`")
                        else:
                            st.info("Tidak ada kombinasi 4D yang memenuhi ambang batas keyakinan.")

with tab_manajemen:
    # ... (Kode tab manajemen tidak berubah) ...
    pass

with tab_scan:
    # ... (Kode tab scan tidak berubah) ...
    pass
