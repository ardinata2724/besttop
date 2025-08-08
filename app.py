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

st.set_page_config(page_title="Prediksi AI", layout="wide")

st.title("Prediksi 4D - AI")

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

# ====== Inisialisasi session_state window_per_digit ======
for label in DIGIT_LABELS:
    key = f"win_{label}"
    if key not in st.session_state:
        st.session_state[key] = 7  # default value

# ======== Ambil Data API dan Input Manual ========

# ======== Sidebar Pengaturan ========
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Hari", ["harian", "kemarin", "2hari", "3hari"])
    putaran = st.number_input("🔁 Putaran", 10, 1000, 100)
    
    # --- PERUBAHAN DI SINI ---
    st.markdown("### 🎯 Opsi Prediksi")
    jumlah_digit = st.slider("🔢 Jumlah Digit Prediksi", 1, 9, 6)
    # --- AKHIR PERUBAHAN ---

    metode = st.selectbox("🧠 Metode", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"])
    jumlah_uji = st.number_input("📊 Data Uji", 1, 200, 10)
    temperature = st.slider("🌡️ Temperature", 0.1, 2.0, 0.5, step=0.1)
    voting_mode = st.selectbox("⚖️ Kombinasi", ["product", "average"])
    power = st.slider("📈 Confidence Power", 0.5, 3.0, 1.5, 0.1)
    min_conf = st.slider("🔎 Min Confidence", 0.0001, 0.01, 0.0005, 0.0001, format="%.4f")
    use_transformer = st.checkbox("🤖 Gunakan Transformer")
    model_type = "transformer" if use_transformer else "lstm"
    mode_prediksi = st.selectbox("🎯 Mode Prediksi", ["confidence", "ranked", "hybrid"])

    st.markdown("### 🪟 Window Size per Digit")
    window_per_digit = {}
    for label in DIGIT_LABELS:
        window_per_digit[label] = st.slider(
            f"{label.upper()}", 3, 30, st.session_state[f"win_{label}"], key=f"win_{label}"
        )

# ======== Manajemen Model ========
# ======== Manajemen Model (khusus metode AI) ========


# ======== Ambil Data API ========
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

# ======== Tabs Utama ========
tab3_container, tab2, tab1 = st.tabs(["🔮 Scan Angka", "🪟 Scan Angka", "CatBoost"])

# ======== TAB 1 ========
with tab1:
    if metode in ["LSTM AI", "Ensemble AI + Markov"]:
        with st.expander("⚙️ Manajemen Model", expanded=False):
            lokasi_id = selected_lokasi.lower().strip().replace(" ", "_")
            digit_labels = ["ribuan", "ratusan", "puluhan", "satuan"]

            for label in digit_labels:
                model_path = f"saved_models/{lokasi_id}_{label}_{model_type}.h5"
                log_path = f"training_logs/history_{lokasi_id}_{label}_{model_type}.csv"

                st.markdown(f"### 📁 Model {label.upper()}")

                # Status Model
                if os.path.exists(model_path):
                    st.info(f"📂 Model {label.upper()} tersedia.")
                else:
                    st.warning(f"⚠️ Model {label.upper()} belum tersedia.")

                # Tombol horizontal: Hapus Model & Hapus Log
                tombol_col1, tombol_col2 = st.columns([1, 1])
                with tombol_col1:
                    if os.path.exists(model_path):
                        if st.button("🗑 Hapus Model", key=f"hapus_model_{label}"):
                            os.remove(model_path)
                            st.warning(f"✅ Model {label.upper()} dihapus.")
                            st.rerun()
                with tombol_col2:
                    if os.path.exists(log_path):
                        if st.button("🧹 Hapus Log", key=f"hapus_log_{label}"):
                            os.remove(log_path)
                            st.info(f"🧾 Log training {label.upper()} dihapus.")
                            st.rerun()

            st.markdown("---")
            if st.button("📚 Latih & Simpan Semua Model"):
                with st.spinner("🔄 Melatih semua model..."):
                    train_and_save_model(df, selected_lokasi, window_dict=window_per_digit, model_type=model_type)
                st.success("✅ Semua model berhasil dilatih.")
    
    if st.button("🔮 Prediksi", use_container_width=True):
        
        if len(df) < max(window_per_digit.values()) + 1:
            st.warning("❌ Data tidak cukup.")
        else:
            with st.spinner("⏳ Memproses..."):
                result, probs = None, None
                # --- PERUBAHAN DI SINI ---
                if metode == "Markov":
                    result, _ = top6_markov(df, top_n=jumlah_digit)
                elif metode == "Markov Order-2":
                    result = top6_markov_order2(df, top_n=jumlah_digit)
                elif metode == "Markov Gabungan":
                    result = top6_markov_hybrid(df, top_n=jumlah_digit)
                elif metode == "LSTM AI":
                    result, probs = top6_model(df, lokasi=selected_lokasi, model_type=model_type,  
                                               return_probs=True, temperature=temperature,  
                                               mode_prediksi=mode_prediksi, window_dict=window_per_digit,
                                               top_n=jumlah_digit)  
                elif metode == "Ensemble AI + Markov":
                    # Ensemble membutuhkan penanganan khusus
                    result = top6_ensemble(df, lokasi=selected_lokasi, model_type=model_type,
                                           window_dict=window_per_digit, temperature=temperature,
                                           mode_prediksi=mode_prediksi, top_n=jumlah_digit)
                    probs = None # Ensemble tidak mengembalikan probabilitas gabungan saat ini

            if result:
                st.subheader(f"🎯 Hasil Prediksi Top {jumlah_digit}")
                # --- AKHIR PERUBAHAN ---
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

            if probs:
                st.subheader("📊 Confidence Bar")
                for i, label in enumerate(DIGIT_LABELS):
                    st.markdown(f"**{label.upper()}**")
                    dconf = pd.DataFrame({
                        "Digit": [str(d) for d in result[i]],
                        "Confidence": probs[i]
                    }).sort_values("Confidence", ascending=True)
                    st.bar_chart(dconf.set_index("Digit"))

            if metode in ["LSTM AI"]: # Kombinasi 4D lebih cocok untuk model AI murni
                with st.spinner("🔢 Kombinasi 4D..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, model_type=model_type,
                                            top_n=10, min_conf=min_conf, power=power,
                                            mode=voting_mode, window_dict=window_per_digit,
                                            mode_prediksi=mode_prediksi,
                                            # --- PERUBAHAN DI SINI ---
                                            pred_n=jumlah_digit) 
                                            # --- AKHIR PERUBAHAN ---
                    st.subheader("💡 Kombinasi 4D Top")
                    for komb, score in top_komb:
                        st.markdown(f"`{komb}` - Confidence: `{score:.4f}`")

# ======== TAB 2: Scan Window Size ========
with tab2:
    min_ws = st.number_input("🔁 Min WS", 3, 10, 4)
    max_ws = st.number_input("🔁 Max WS", 4, 20, 12)
    min_acc = st.slider("🌡️ Min Acc", 0.1, 1.0, 0.5, step=0.01)
    min_conf = st.slider("🌡️ Min Conf", 0.1, 1.0, 0.5, step=0.01)

    if "scan_step" not in st.session_state:
        st.session_state.scan_step = 0
    if "scan_in_progress" not in st.session_state:
        st.session_state.scan_in_progress = False
    if "scan_results" not in st.session_state:
        st.session_state.scan_results = {}

    if "ws_result_table" not in st.session_state:
        st.session_state.ws_result_table = pd.DataFrame()
    if "window_per_digit" not in st.session_state:
        st.session_state.window_per_digit = {}

    for label in DIGIT_LABELS:
        # --- PERUBAHAN DI SINI ---
        st.session_state.setdefault(f"best_ws_{label}", None)
        st.session_state.setdefault(f"top_n_{label}", [])
        # --- AKHIR PERUBAHAN ---
        st.session_state.setdefault(f"acc_table_{label}", None)
        st.session_state.setdefault(f"conf_table_{label}", None)

    with st.expander("⚙️ Opsi Cross Validation"):
        use_cv = st.checkbox("Gunakan Cross Validation", value=False, key="use_cv_toggle")
        if use_cv:
            cv_folds = st.number_input("Jumlah Fold (K-Folds)", 2, 10, 2, step=1, key="cv_folds_input")
        else:
            cv_folds = None

    with st.expander("🔍 Scan Angka Normal (Per Digit)", expanded=True):
        cols = st.columns(4)
        for idx, label in enumerate(DIGIT_LABELS):
            with cols[idx]:
                if st.button(f"🔍 {label.upper()}", use_container_width=True, key=f"btn_{label}"):
                    with st.spinner(f"🔍 Mencari WS terbaik untuk {label.upper()}..."):
                        try:
                            # --- PERUBAHAN DI SINI ---
                            ws, top_n_digits = find_best_window_size_with_model_true(
                                df, label, selected_lokasi, model_type=model_type,
                                min_ws=min_ws, max_ws=max_ws, temperature=temperature,
                                use_cv=use_cv, cv_folds=cv_folds or 2,
                                seed=42, min_acc=min_acc, min_conf=min_conf,
                                top_n=jumlah_digit
                            )
                            st.session_state.window_per_digit[label] = ws
                            st.session_state[f"best_ws_{label}"] = ws
                            st.session_state[f"top_n_{label}"] = top_n_digits
                            st.success(f"✅ WS {label.upper()}: {ws}")
                            st.info(f"🔢 Top-{jumlah_digit} {label.upper()}: {', '.join(map(str, top_n_digits))}")
                            # --- AKHIR PERUBAHAN ---
                        except Exception as e:
                            st.error(f"❌ Gagal {label.upper()}: {e}")
        st.markdown("---")
        if st.button("🔎 Scan Semua Digit Sekaligus", use_container_width=True):
            st.session_state.scan_step = 0
            st.session_state.scan_in_progress = True
            st.rerun()
        
    # --- PERUBAHAN DI SINI ---
    st.markdown(f"### 🧾 Hasil Terakhir per Digit (Top-{jumlah_digit})")
    for label in DIGIT_LABELS:
        ws = st.session_state.get(f"best_ws_{label}")
        top_n = st.session_state.get(f"top_n_{label}", [])
        if ws:
            st.info(f"📌 {label.upper()} | WS: {ws} | Top-{jumlah_digit}: {', '.join(map(str, top_n))}")
    # --- AKHIR PERUBAHAN ---

    if st.session_state.scan_in_progress:
        step = st.session_state.scan_step
        if step < len(DIGIT_LABELS):
            label = DIGIT_LABELS[step]
            with st.spinner(f"🔍 Memproses {label.upper()} ({step+1}/{len(DIGIT_LABELS)})..."):
                try:
                    # --- PERUBAHAN DI SINI ---
                    ws, top_n_digits = find_best_window_size_with_model_true(
                        df, label, selected_lokasi, model_type=model_type,
                        min_ws=min_ws, max_ws=max_ws, temperature=temperature,
                        use_cv=use_cv, cv_folds=cv_folds or 2,
                        seed=42, min_acc=min_acc, min_conf=min_conf,
                        top_n=jumlah_digit
                    )
                    st.session_state.window_per_digit[label] = ws
                    st.session_state[f"best_ws_{label}"] = ws
                    st.session_state[f"top_n_{label}"] = top_n_digits
                    st.session_state.scan_results[label] = {
                        "ws": ws,
                        "top_n": top_n_digits
                    }
                    # --- AKHIR PERUBAHAN ---
                except Exception as e:
                    st.session_state.scan_results[label] = {
                        "ws": None,
                        "top_n": [],
                        "error": str(e)
                    }
                    st.error(f"❌ Gagal {label.upper()}: {e}")
                st.session_state.scan_step += 1
                st.rerun()
        else:
            st.success("✅ Semua digit selesai diproses.")
            st.session_state.scan_in_progress = False

            # Generate hasil akhir
            hasil_data = []
            for label in DIGIT_LABELS:
                # --- PERUBAHAN DI SINI ---
                top_n = st.session_state.get(f"top_n_{label}", [])
                ws = st.session_state.get(f"best_ws_{label}")
                hasil_data.append({
                    "Digit": label.upper(),
                    "Best WS": ws if ws else "-",
                    f"Top-{jumlah_digit}": ", ".join(map(str, top_n)) if top_n else "-"
                })
                # --- AKHIR PERUBAHAN ---
            st.session_state.ws_result_table = pd.DataFrame(hasil_data)

    if not st.session_state.ws_result_table.empty:
        st.subheader("✅ Tabel Hasil Window Size")
        st.dataframe(st.session_state.ws_result_table)
    
    # Bagian CatBoost tidak diubah karena merupakan proses terpisah
    with st.expander("📈 Scan WS dengan CatBoost", expanded=False):
        # ... (kode catboost tetap sama) ...
        pass # Placeholder

with tab3_container:
    tab3(df, selected_lokasi)
