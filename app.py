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
    scan_ws_catboost,  # Pastikan file ini ada dan telah ter-import
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

    #st.subheader("📊 Evaluasi Akurasi")
    #acc1, acc6, top1 = evaluate_lstm_accuracy_all_digits(
    #    df, selected_lokasi, model_type=model_type, window_size=window_per_digit
    #)
    #for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
    #    st.info(f"🎯 {label}: Top-1 = {acc1[i]:.2%}, Top-6 = {acc6[i]:.2%}")

# ======== TAB 2 ========
# ======== TAB 2: Scan Window Size ========
with tab2:
    min_ws = st.number_input("🔁 Min WS", 3, 10, 4)
    max_ws = st.number_input("🔁 Max WS", 4, 20, 12)
    min_acc = st.slider("🌡️ Min Acc", 0.1, 2.0, 0.5, step=0.1)
    min_conf = st.slider("🌡️ Min Conf", 0.1, 2.0, 0.5, step=0.1)

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
                                seed=42, min_acc=min_acc, min_conf=min_conf
                            )
                            st.session_state.window_per_digit[label] = ws
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
        
    st.markdown("### 🧾 Hasil Terakhir per Digit")
    for label in DIGIT_LABELS:
        ws = st.session_state.get(f"best_ws_{label}")
        top6 = st.session_state.get(f"top6_{label}", [])
        if ws:
            st.info(f"📌 {label.upper()} | WS: {ws} | Top-6: {', '.join(map(str, top6))}")

    

    

    if st.session_state.scan_in_progress:
        step = st.session_state.scan_step
        if step < len(DIGIT_LABELS):
            label = DIGIT_LABELS[step]
            with st.spinner(f"🔍 Memproses {label.upper()} ({step+1}/{len(DIGIT_LABELS)})..."):
                try:
                    ws, top6 = find_best_window_size_with_model_true(
                        df, label, selected_lokasi, model_type=model_type,
                        min_ws=min_ws, max_ws=max_ws, temperature=temperature,
                        use_cv=use_cv, cv_folds=cv_folds or 2,
                        seed=42, min_acc=min_acc, min_conf=min_conf
                    )
                    st.session_state.window_per_digit[label] = ws
                    st.session_state[f"best_ws_{label}"] = ws
                    st.session_state[f"top6_{label}"] = top6
                    st.session_state.scan_results[label] = {
                        "ws": ws,
                        "top6": top6
                    }
                except Exception as e:
                    st.session_state.scan_results[label] = {
                        "ws": None,
                        "top6": [],
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
                top6 = st.session_state.get(f"top6_{label}", [])
                ws = st.session_state.get(f"best_ws_{label}")
                hasil_data.append({
                    "Digit": label.upper(),
                    "Best WS": ws if ws else "-",
                    "Top6": ", ".join(map(str, top6)) if top6 else "-"
                })
            st.session_state.ws_result_table = pd.DataFrame(hasil_data)

    if not st.session_state.ws_result_table.empty:
        st.subheader("✅ Tabel Hasil Window Size")
        st.dataframe(st.session_state.ws_result_table)

        try:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis('off')
            tbl = ax.table(
                cellText=st.session_state.ws_result_table.values,
                colLabels=st.session_state.ws_result_table.columns,
                cellLoc='center',
                loc='center'
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.5)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Gagal tampilkan tabel: {e}")

    # Tambahkan heatmap untuk setiap digit jika tersedia
    for label in DIGIT_LABELS:
        acc_df = st.session_state.get(f"acc_table_{label}")
        conf_df = st.session_state.get(f"conf_table_{label}")
        if acc_df is not None:
            st.markdown(f"#### 🔥 Heatmap Akurasi - {label.upper()}")
            fig1, ax1 = plt.subplots(figsize=(8, 1.5))
            sns.heatmap(acc_df.T, annot=True, cmap="YlGnBu", cbar=False, ax=ax1)
            st.pyplot(fig1)
        if conf_df is not None:
            st.markdown(f"#### 🔥 Heatmap Confidence - {label.upper()}")
            fig2, ax2 = plt.subplots(figsize=(8, 1.5))
            sns.heatmap(conf_df.T, annot=True, cmap="Oranges", cbar=False, ax=ax2)
            st.pyplot(fig2)
            
    with st.expander("📈 Scan WS dengan CatBoost", expanded=False):
        selected_digit = st.selectbox("📌 Pilih Digit", DIGIT_LABELS, key="catboost_digit")
        min_ws_cb = st.number_input("🔁 Min WS (CatBoost)", 3, 30, 5, key="cb_min_ws")
        max_ws_cb = st.number_input("🔁 Max WS (CatBoost)", min_ws_cb + 1, 50, 15, key="cb_max_ws")
        folds_cb = st.slider("📂 Jumlah Fold (CV)", 2, 10, 3, key="cb_folds")

        if "catboost_result" not in st.session_state:
            st.session_state.catboost_result = {}
        if "catboost_best_ws" not in st.session_state:
            st.session_state.catboost_best_ws = {}

        if st.button("🔍 Scan CatBoost (Semua Digit)", use_container_width=True):
            st.subheader("⏳ Proses Scan Window Size dengan CatBoost")
            progress_bar = st.progress(0.0, text="Memulai...")

            for idx, label in enumerate(DIGIT_LABELS):
                progress_text = f"🔄 Memproses {label.upper()} ({idx+1}/{len(DIGIT_LABELS)})..."
                progress_bar.progress(idx / len(DIGIT_LABELS), text=progress_text)
                try:
                    result_df = scan_ws_catboost(df, label, min_ws=min_ws_cb, max_ws=max_ws_cb, cv_folds=folds_cb, seed=42)
                    st.session_state.catboost_result[label] = result_df

                    if not result_df.empty:
                        best_row = result_df.loc[result_df["Accuracy Mean"].idxmax()]
                        st.session_state.catboost_best_ws[label] = int(best_row["WS"])
                        st.success(f"✅ {label.upper()}: WS terbaik = {int(best_row['WS'])} | Akurasi: {best_row['Accuracy Mean']:.2%}")
                    else:
                        st.warning(f"⚠️ Tidak ada hasil untuk {label.upper()}")

                except Exception as e:
                    st.session_state.catboost_result[label] = None
                    st.error(f"❌ Gagal proses {label.upper()}: {e}")

            progress_bar.progress(1.0, text="✅ Selesai")
            st.success("🎉 Semua digit selesai diproses dengan CatBoost.")

        # Tampilkan hasil
        if st.session_state.catboost_result:
            st.subheader("📊 Hasil CatBoost per Digit")

            for label in DIGIT_LABELS:
                result = st.session_state.catboost_result.get(label)

                if result is None or isinstance(result, str):
                    st.error(f"❌ {label.upper()}: Gagal atau kosong")
                    continue

                st.markdown(f"### 📍 {label.upper()}")
                st.dataframe(result.round(4), use_container_width=True)

                # Tampilkan WS terbaik
                best_ws = st.session_state.catboost_best_ws.get(label)
                if best_ws:
                    st.info(f"✅ Window Size terbaik: `{best_ws}`")

                # Visualisasi bar chart
                try:
                    fig, ax = plt.subplots(figsize=(7, 3))
                    ax.bar(result["WS"], result["Accuracy Mean"], color="skyblue")
                    ax.set_title(f"Akurasi vs WS - {label.upper()}")
                    ax.set_xlabel("Window Size")
                    ax.set_ylabel("Accuracy Mean")
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"⚠️ Gagal visualisasi: {e}")

with tab3_container:
    tab3(df, selected_lokasi)
#with tab4_container:
#    tab4(df)
#with tab5_container:
#    tab6(df, selected_lokasi)
            
