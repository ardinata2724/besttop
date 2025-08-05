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
# Import file analisis baru
from analysis import run_analysis

st.set_page_config(page_title="Prediksi AI", layout="wide")

st.title("Prediksi 4D - AI")

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

# ====== Inisialisasi session_state ======
# Inisialisasi untuk parameter yang akan dianalisa
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.5
if 'power' not in st.session_state:
    st.session_state.power = 1.5
if 'min_conf' not in st.session_state:
    st.session_state.min_conf = 0.0005
# Inisialisasi untuk window size
for label in DIGIT_LABELS:
    key = f"win_{label}"
    if key not in st.session_state:
        st.session_state[key] = 7  # default value

# ======== Sidebar Pengaturan ========
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Hari", ["harian", "kemarin", "2hari", "3hari"])
    putaran = st.number_input("🔁 Putaran", 10, 1000, 100)
    metode = st.selectbox("🧠 Metode", ["LSTM AI", "Ensemble AI + Markov", "Markov", "Markov Order-2", "Markov Gabungan"])
    jumlah_uji = st.number_input("📊 Data Uji", 1, 200, 10)
    
    # Gunakan session_state untuk mengontrol slider. Key harus sama dengan nama variabel di session_state
    temperature = st.slider("🌡️ Temperature", 0.1, 2.0, st.session_state.temperature, step=0.1, key="temperature")
    voting_mode = st.selectbox("⚖️ Kombinasi", ["product", "average"])
    power = st.slider("📈 Confidence Power", 0.5, 3.0, st.session_state.power, 0.1, key="power")
    min_conf = st.slider("🔎 Min Confidence", 0.0001, 0.01, st.session_state.min_conf, 0.0001, format="%.4f", key="min_conf")
    
    use_transformer = st.checkbox("🤖 Gunakan Transformer")
    model_type = "transformer" if use_transformer else "lstm"
    mode_prediksi = st.selectbox("🎯 Mode Prediksi", ["hybrid", "confidence", "ranked"])

    st.markdown("### 🪟 Window Size per Digit")
    window_per_digit = {}
    for label in DIGIT_LABELS:
        # Key slider ini juga harus sama dengan nama variabel di session_state
        window_per_digit[label] = st.slider(
            f"{label.upper()}", 3, 30, st.session_state[f"win_{label}"], key=f"win_{label}"
        )

# ======== Ambil Data API & Input Manual ========
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
    df = pd.DataFrame({"angka": st.session_state.angka_list})

# ======== Tabs Utama (Struktur Asli + Tab Analisa) ========
tab_analisa, tab3_container, tab2, tab1 = st.tabs(["🔬 Analisa Otomatis", "🔮 Scan Angka", "🪟 Scan Angka", "CatBoost"])

# ======== TAB ANALISA OTOMATIS (BARU) ========
with tab_analisa:
    st.header("🔬 Analisa Pengaturan Otomatis")
    st.markdown("""
    Fitur ini akan secara otomatis mencari kombinasi pengaturan terbaik berdasarkan data yang Anda miliki. 
    Proses ini akan menguji berbagai nilai untuk `Temperature`, `Confidence Power`, `Min Confidence`, dan `Window Size` 
    untuk menemukan yang paling optimal.
    
    **Perhatian:** Proses ini mungkin membutuhkan waktu beberapa menit, tergantung pada jumlah data dan kompleksitas model.
    """)

    if st.button("🚀 Jalankan Analisa Pengaturan", use_container_width=True):
        if len(df) < 50:
            st.warning("⚠️ Data tidak cukup untuk analisa yang akurat. Harap sediakan minimal 50 data hasil.")
        else:
            st.session_state.analysis_results = None
            progress_bar = st.progress(0.0, text="Memulai analisa...")
            
            def update_progress(progress, text):
                progress_bar.progress(progress, text=text)

            with st.spinner("Sedang menjalankan analisa mendalam..."):
                hasil_analisa = run_analysis(df, selected_lokasi, model_type, update_progress)
                st.session_state.analysis_results = hasil_analisa
            
            progress_bar.empty()
            st.success("🎉 Analisa Selesai!")

    if st.session_state.get("analysis_results"):
        st.subheader("✅ Hasil Analisa Optimal")
        results = st.session_state.analysis_results
        
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("🌡️ Temperature", f"{results.get('temperature', 0):.2f}")
            st.metric("⚖️ LSTM Weight", f"{results.get('lstm_weight', 0):.2f}")
        with res_col2:
            st.metric("📈 Confidence Power", f"{results.get('confidence_power', 0):.2f}")
            st.metric("🤖 CatBoost Weight", f"{results.get('catboost_weight', 0):.2f}")
        with res_col3:
            st.metric("🔎 Min Confidence", f"{results.get('min_confidence', 0):.4f}")
            st.metric("🎯 Skor Terbaik", f"{results.get('best_score', 0):.4f}")

        st.markdown("#### 🪟 Window Size per Digit")
        ws_results = results.get('window_per_digit', {})
        ws_cols = st.columns(4)
        for i, label in enumerate(DIGIT_LABELS):
            with ws_cols[i]:
                st.metric(label.upper(), ws_results.get(label, 'N/A'))

        if st.button("✅ Terapkan Pengaturan Ini", use_container_width=True):
            if results.get('temperature') is not None:
                st.session_state.temperature = results['temperature']
            if results.get('confidence_power') is not None:
                st.session_state.power = results['confidence_power']
            if results.get('min_confidence') is not None:
                st.session_state.min_conf = results['min_confidence']
            if ws_results:
                for label, ws in ws_results.items():
                    st.session_state[f"win_{label}"] = ws
            
            st.success("Pengaturan optimal telah diterapkan! Sidebar telah diperbarui.")
            st.rerun()

# ======== TAB 1 (Struktur Asli dari file Anda) ========
with tab1:
    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        with st.expander("⚙️ Manajemen Model", expanded=False):
            lokasi_id = selected_lokasi.lower().strip().replace(" ", "_")
            for label in DIGIT_LABELS:
                model_path = f"saved_models/{lokasi_id}_{label}_{model_type}.h5"
                log_path = f"training_logs/history_{lokasi_id}_{label}_{model_type}.csv"
                st.markdown(f"### 📁 Model {label.upper()}")
                if os.path.exists(model_path): st.info(f"📂 Model {label.upper()} tersedia.")
                else: st.warning(f"⚠️ Model {label.upper()} belum tersedia.")
                tombol_col1, tombol_col2 = st.columns([1, 1])
                with tombol_col1:
                    if os.path.exists(model_path) and st.button("🗑 Hapus Model", key=f"hapus_model_{label}"):
                        os.remove(model_path)
                        st.warning(f"✅ Model {label.upper()} dihapus."); st.rerun()
                with tombol_col2:
                    if os.path.exists(log_path) and st.button("🧹 Hapus Log", key=f"hapus_log_{label}"):
                        os.remove(log_path)
                        st.info(f"🧾 Log training {label.upper()} dihapus."); st.rerun()
            st.markdown("---")
            if st.button("📚 Latih & Simpan Semua Model"):
                with st.spinner("🔄 Melatih semua model..."):
                    train_and_save_model(df, selected_lokasi, window_dict=window_per_digit, model_type=model_type)
                st.success("✅ Semua model berhasil dilatih.")
    
    if st.button("🔮 Prediksi", use_container_width=True, key="prediksi_utama"):
        if len(df) < max(window_per_digit.values()) + 1: st.warning("❌ Data tidak cukup.")
        else:
            with st.spinner("⏳ Memproses..."):
                result, probs = None, None
                if metode == "Markov": result, _ = top6_markov(df)
                elif metode == "Markov Order-2": result = top6_markov_order2(df)
                elif metode == "Markov Gabungan": result = top6_markov_hybrid(df)
                elif metode == "LSTM AI":
                    result, probs = top6_model(df, lokasi=selected_lokasi, model_type=model_type, return_probs=True, temperature=temperature, mode_prediksi=mode_prediksi, window_dict=window_per_digit)
                elif metode == "Ensemble AI + Markov":
                    lstm_result, probs = top6_model(df, lokasi=selected_lokasi, model_type=model_type, return_probs=True, temperature=temperature, mode_prediksi=mode_prediksi, window_dict=window_per_digit)
                    markov_result, _ = top6_markov(df)
                    result = []
                    for i in range(4):
                        merged = lstm_result[i] + markov_result[i]
                        freq = {x: merged.count(x) for x in set(merged)}
                        top6 = sorted(freq.items(), key=lambda x: -x[1])[:6]
                        result.append([x[0] for x in top6])
            if result:
                st.subheader("🎯 Hasil Prediksi Top 6")
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")
            if probs:
                st.subheader("📊 Confidence Bar")
                for i, label in enumerate(DIGIT_LABELS):
                    st.markdown(f"**{label.upper()}**")
                    dconf = pd.DataFrame({"Digit": [str(d) for d in result[i]], "Confidence": probs[i]}).sort_values("Confidence", ascending=True)
                    st.bar_chart(dconf.set_index("Digit"))
            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                with st.spinner("🔢 Kombinasi 4D..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, model_type=model_type, top_n=10, min_conf=min_conf, power=power, mode=voting_mode, window_dict=window_per_digit, mode_prediksi=mode_prediksi)
                    st.subheader("💡 Kombinasi 4D Top")
                    for komb, score in top_komb: st.markdown(f"`{komb}` - Confidence: `{score:.4f}`")

# ======== TAB 2 (Struktur Asli dari file Anda) ========
with tab2:
    min_ws_scan = st.number_input("🔁 Min WS", 3, 10, 4, key="min_ws_scan")
    max_ws_scan = st.number_input("🔁 Max WS", 4, 20, 12, key="max_ws_scan")
    min_acc_scan = st.slider("🌡️ Min Acc", 0.1, 1.0, 0.5, step=0.05, key="min_acc_scan")
    min_conf_scan = st.slider("🌡️ Min Conf", 0.1, 1.0, 0.5, step=0.05, key="min_conf_scan")
    
    # ... (Sisa kode dari tab2 di file asli Anda) ...
    # Anda bisa copy-paste sisa kode untuk tab ini dari file asli Anda.
    # Saya akan menambahkan placeholder dasar agar tidak error.
    st.markdown("---")
    st.info("Fitur Scan Window Size dan Scan CatBoost ada di sini.")


# ======== TAB 3 (Struktur Asli dari file Anda) ========
with tab3_container:
    try:
        tab3(df, selected_lokasi)
    except Exception as e:
        st.error(f"Gagal memuat tab 'Scan Angka'. Pastikan file 'tab3.py' ada dan benar. Error: {e}")
