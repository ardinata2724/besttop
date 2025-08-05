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

# --- Pengaturan Preset untuk Menu Analisa ---
# Setelan spesifik dikosongkan sesuai permintaan. Semua pasaran akan menggunakan setelan default.
PRESET_SETTINGS = {}

# Setelan default untuk semua pasaran yang dipilih di menu Analisa
DEFAULT_ANALYZE_PRESET = {
    "temperature": 1.0, "voting_mode": "product", "power": 1.7, "min_conf": 0.0030,
    "mode_prediksi": "hybrid", "win_ribuan": 9, "win_ratusan": 9, "win_puluhan": 9, "win_satuan": 9,
}


# --- Inisialisasi dan Fungsi Callback untuk Session State ---
def initialize_state():
    defaults = {
        "temperature": 0.5, "voting_mode": "product", "power": 1.5, "min_conf": 0.0005,
        "mode_prediksi": "confidence", "win_ribuan": 7, "win_ratusan": 7, "win_puluhan": 7, "win_satuan": 7,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def apply_preset():
    preset_key = st.session_state.analisa_choice
    if preset_key != "Tidak Ada":
        # Karena PRESET_SETTINGS kosong, .get() akan selalu mengembalikan DEFAULT_ANALYZE_PRESET
        settings = PRESET_SETTINGS.get(preset_key, DEFAULT_ANALYZE_PRESET)
        for key, value in settings.items():
            st.session_state[key] = value

# Panggil inisialisasi di awal
initialize_state()

# ======== Sidebar Pengaturan ========
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Hari", ["harian", "kemarin", "2hari", "3hari"])
    putaran = st.number_input("🔁 Putaran", 10, 1000, 100)
    metode = st.selectbox("🧠 Metode", ["Markov", "Markov Order-2", "Markov Gabungan", "LSTM AI", "Ensemble AI + Markov"])
    jumlah_uji = st.number_input("📊 Data Uji", 1, 200, 10)
    
    # --- Menu Analisa dinamis berdasarkan semua pasaran ---
    st.selectbox(
        "📈 Analisa",
        ["Tidak Ada"] + lokasi_list,  # Menggunakan `lokasi_list` untuk pilihan
        key="analisa_choice",
        on_change=apply_preset,
        help="Pilih pasaran untuk menerapkan setelan otomatis."
    )
    
    # --- Widget yang terhubung dengan session state ---
    temperature = st.slider("🌡️ Temperature", 0.1, 2.0, key="temperature", step=0.1)
    voting_mode = st.selectbox("⚖️ Kombinasi", ["product", "average"], key="voting_mode")
    power = st.slider("📈 Confidence Power", 0.5, 3.0, key="power", step=0.1)
    min_conf = st.slider("🔎 Min Confidence", 0.0001, 0.01, key="min_conf", format="%.4f")
    use_transformer = st.checkbox("🤖 Gunakan Transformer")
    model_type = "transformer" if use_transformer else "lstm"
    mode_prediksi = st.selectbox("🎯 Mode Prediksi", ["confidence", "ranked", "hybrid"], key="mode_prediksi")

    st.markdown("### 🪟 Window Size per Digit")
    window_per_digit = {}
    for label in DIGIT_LABELS:
        window_per_digit[label] = st.slider(
            f"{label.upper()}", 3, 30, key=f"win_{label}"
        )

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

                if os.path.exists(model_path):
                    st.info(f"📂 Model {label.upper()} tersedia.")
                else:
                    st.warning(f"⚠️ Model {label.upper()} belum tersedia.")

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
                if metode == "Markov":
                    result, _ = top6_markov(df)
                elif metode == "Markov Order-2":
                    result = top6_markov_order2(df)
                elif metode == "Markov Gabungan":
                    result = top6_markov_hybrid(df)
                elif metode == "LSTM AI":
                    result, probs = top6_model(df, lokasi=selected_lokasi, model_type=model_type,  
                                               return_probs=True, temperature=temperature,  
                                               mode_prediksi=mode_prediksi, window_dict=window_per_digit)  
                elif metode == "Ensemble AI + Markov":
                    lstm_result, probs = top6_model(df, lokasi=selected_lokasi, model_type=model_type,  
                                                    return_probs=True, temperature=temperature,  
                                                    mode_prediksi=mode_prediksi, window_dict=window_per_digit)  
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
                    dconf = pd.DataFrame({
                        "Digit": [str(d) for d in result[i]],
                        "Confidence": probs[i]
                    }).sort_values("Confidence", ascending=True)
                    st.bar_chart(dconf.set_index("Digit"))

            if metode in ["LSTM AI", "Ensemble AI + Markov"]:
                with st.spinner("🔢 Kombinasi 4D..."):
                    top_komb = kombinasi_4d(df, lokasi=selected_lokasi, model_type=model_type,
                                            top_n=10, min_conf=min_conf, power=power,
                                            mode=voting_mode, window_dict=window_per_digit,
                                            mode_prediksi=mode_prediksi)
                    st.subheader("💡 Kombinasi 4D Top")
                    for komb, score in top_komb:
                        st.markdown(f"`{komb}` - Confidence: `{score:.4f}`")

# ======== Sisa kode (TAB 2, TAB 3, dst.) tetap sama ========
with tab2:
    min_ws = st.number_input("🔁 Min WS", 3, 10, 4)
    max_ws = st.number_input("🔁 Max WS", 4, 20, 12)
    min_acc_slider = st.slider("🌡️ Min Acc", 0.1, 1.0, 0.6, step=0.05, key="min_acc_slider_tab2")
    min_conf_slider = st.slider("🌡️ Min Conf", 0.1, 1.0, 0.6, step=0.05, key="min_conf_slider_tab2")

    if "scan_step" not in st.session_state:
        st.session_state.scan_step = 0
    if "scan_in_progress" not in st.session_state:
        st.session_state.scan_in_progress = False
    if "scan_results" not in st.session_state:
        st.session_state.scan_results = {}

    if "ws_result_table" not in st.session_state:
        st.session_state.ws_result_table = pd.DataFrame()
    
    for label in DIGIT_LABELS:
        st.session_state.setdefault(f"best_ws_{label}", None)
        st.session_state.setdefault(f"top6_{label}", [])
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
                            ws, top6 = find_best_window_size_with_model_true(
                                df, label, selected_lokasi, model_type=model_type,
                                min_ws=min_ws, max_ws=max_ws, temperature=temperature,
                                use_cv=use_cv, cv_folds=cv_folds or 2,
                                seed=42, min_acc=min_acc_slider, min_conf=min_conf_slider
                            )
                            st.session_state[f"best_ws_{label}"] = ws
                            st.session_state[f"top6_{label}"] = top6
                            st.success(f"✅ WS {label.upper()}: {ws}")
                            st.info(f"🔢 Top-6 {label.upper()}: {', '.join(map(str, top6))}")
                        except Exception as e:
                            st.error(f"❌ Gagal {label.upper()}: {e}")
        st.markdown("---")
        if st.button("🔎 Scan Semua Digit Sekaligus", use_container_width=True):
            st.session_state.scan_step = 0
            st.session_state.scan_in_progress = True
            st.rerun()

# ... (sisa kode untuk tab lainnya tidak diubah) ...
with tab3_container:
    tab3(df, selected_lokasi)
