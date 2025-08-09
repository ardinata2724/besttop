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
                    pred_data = top6_model(df, lokasi=selected_lokasi, model_type=model_type, return_probs=True, temperature=temperature, mode_prediksi=mode_prediksi, window_dict=window_per_digit, top_n=jumlah_digit)
                    if pred_data: result, probs = pred_data
                    else: st.error("Gagal memuat model AI. Pastikan model sudah dilatih.")
                
                elif metode == "Ensemble AI + Markov":
                    result = top6_ensemble(df, lokasi=selected_lokasi, model_type=model_type, window_dict=window_per_digit, temperature=temperature, mode_prediksi=mode_prediksi, top_n=jumlah_digit)
                    if result is None: st.error("Gagal prediksi ensemble. Pastikan model AI sudah dilatih.")
            
            if result:
                st.subheader(f"🎯 Hasil Prediksi Top {jumlah_digit}")
                for i, label in enumerate(["Ribuan", "Ratusan", "Puluhan", "Satuan"]):
                    st.markdown(f"**{label}:** {', '.join(map(str, result[i]))}")

                st.divider()
                st.subheader("🎲 Acak 4D dari Hasil Prediksi (Sistem Rotasi)")
                if all(result) and len(result) == 4:
                    try:
                        ribuan_list, ratusan_list, puluhan_list, satuan_list = result[0], result[1], result[2], result[3]
                        patterns = [(ribuan_list, ratusan_list, puluhan_list, satuan_list), (ratusan_list, puluhan_list, satuan_list, ribuan_list), (puluhan_list, satuan_list, ribuan_list, ratusan_list), (satuan_list, ribuan_list, ratusan_list, puluhan_list)]
                        acak_4d_list = []
                        for _ in range(1000):
                            chosen_pattern = random.choice(patterns)
                            d1, d2, d3, d4 = random.choice(chosen_pattern[0]), random.choice(chosen_pattern[1]), random.choice(chosen_pattern[2]), random.choice(chosen_pattern[3])
                            acak_4d_list.append(f"{d1}{d2}{d3}{d4}")
                        output_string = " * ".join(acak_4d_list)
                        st.text_area(f"1000 Kombinasi Acak (Pola Rotasi)", output_string, height=300)
                    except IndexError:
                        st.error("Gagal menghasilkan angka acak, pastikan setiap posisi digit memiliki hasil prediksi.")
                else:
                    st.warning("Tidak bisa menghasilkan angka acak karena salah satu hasil prediksi kosong.")

with tab_manajemen:
    st.subheader("Manajemen Model AI")
    st.info("Latih model AI di sini. Jika Anda mengubah pengaturan window size, Anda harus melatih ulang model.")
    lokasi_id = selected_lokasi.lower().strip().replace(" ", "_")
    cols = st.columns(4)
    for i, label in enumerate(DIGIT_LABELS):
        with cols[i]:
            model_path = f"saved_models/{lokasi_id}_{label}_{model_type}.h5"
            st.markdown(f"##### {label.upper()}")
            if os.path.exists(model_path):
                st.success("✅ Tersedia")
                if st.button("Hapus", key=f"hapus_{label}", use_container_width=True):
                    os.remove(model_path); st.rerun()
            else:
                st.warning("⚠️ Belum ada")
    st.markdown("---")
    if st.button("📚 Latih & Simpan Semua Model AI", use_container_width=True, type="primary"):
        max_ws_needed = max(list(window_per_digit.values()))
        if len(df) < max_ws_needed + 5:
            st.error(f"Data tidak cukup untuk melatih. Dibutuhkan setidaknya {max_ws_needed + 5} baris data.")
        else:
            with st.spinner("🔄 Melatih semua model, ini mungkin memakan waktu..."):
                train_and_save_model(df, selected_lokasi, window_dict=window_per_digit, model_type=model_type)
            st.success("✅ Semua model berhasil dilatih dan disimpan."); st.rerun()

with tab_scan:
    st.subheader("Pencarian Window Size Optimal")
    st.info("Jalankan scan untuk setiap digit guna menemukan Window Size (WS) terbaik. Hasilnya akan ditampilkan di bawah, kemudian atur slider di sidebar secara manual.")

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
                    # Panggil fungsi find_best_window_size_with_model_true
                    # Fungsi ini akan menampilkan tabel dan hasilnya sendiri di dalam expander
                    find_best_window_size_with_model_true(
                        df, label, selected_lokasi, model_type=model_type,
                        min_ws=min_ws, max_ws=max_ws, temperature=temperature,
                        top_n=jumlah_digit, min_acc=min_acc, min_conf=min_conf
                    )
