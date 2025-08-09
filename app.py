import streamlit as st
import pandas as pd
import requests
import os
import time
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter
from itertools import product
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dropout, Dense,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from datetime import datetime

# ==============================================================================
# BAGIAN 1: DEFINISI FUNGSI-FUNGSI INTI
# (Tidak ada perubahan di bagian ini, semua fungsi tetap sama)
# ==============================================================================
DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

@st.cache_data(ttl=600)
def fetch_live_results(pasaran_list):
    results = []
    headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
    for pasaran in pasaran_list:
        try:
            pasaran_id = pasaran.lower().replace(' ', '')
            url = f"https://wysiwygscan.com/api?pasaran={pasaran_id}&hari=harian&putaran=1&format=json&urut=desc"
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("data") and data["data"]:
                latest_result = data["data"][0]
                tanggal = datetime.strptime(latest_result['tanggal'], '%Y-%m-%d').strftime('%d-%m-%Y')
                results.append({"Pasaran": pasaran.title(), "Tanggal": tanggal, "Hasil": latest_result["result"]})
            else:
                results.append({"Pasaran": pasaran.title(), "Tanggal": "-", "Hasil": "..."})
        except Exception:
            results.append({"Pasaran": pasaran.title(), "Tanggal": "Error", "Hasil": "Gagal"})
    return results

def _ensure_unique_top_n(top_list, n=6):
    unique_list = list(dict.fromkeys(top_list))[:n]
    if len(unique_list) >= n: return unique_list
    all_digits = list(range(10)); random.shuffle(all_digits)
    unique_set = set(unique_list)
    for digit in all_digits:
        if len(unique_set) >= n: break
        if digit not in unique_set: unique_set.add(digit)
    return list(unique_set)

def top6_markov(df, top_n=6):
    if df.empty or len(df) < 10: return [], None
    data = df["angka"].astype(str).tolist()
    matrix = [defaultdict(lambda: defaultdict(int)) for _ in range(3)]
    for number in data:
        digits = f"{int(number):04d}"
        for i in range(3): matrix[i][digits[i]][digits[i+1]] += 1
    freq_ribuan = Counter([int(x[0]) for x in data])
    hasil = [[k for k, _ in freq_ribuan.most_common(top_n)]]
    for i in range(3):
        kandidat = [int(k) for prev in matrix[i] for k in matrix[i][prev].keys()]
        top = [k for k, _ in Counter(kandidat).most_common()]
        hasil.append(top)
    return [_ensure_unique_top_n(h, n=top_n) for h in hasil], None

class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, x):
        seq_len, d_model = tf.shape(x)[1], tf.shape(x)[2]
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], dtype=tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], dtype=tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines, cosines = tf.math.sin(angle_rads[:, 0::2]), tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return x + tf.cast(tf.expand_dims(pos_encoding, 0), tf.float32)

# ... (sisa fungsi inti tidak berubah) ...


# ==============================================================================
# BAGIAN 2: APLIKASI STREAMLIT UTAMA
# ==============================================================================

st.set_page_config(page_title="Prediksi AI", layout="wide")
st.title("Prediksi 4D - AI")

try: from lokasi_list import lokasi_list
except ImportError: lokasi_list = ["Bullseye", "California", "Cambodia", "China", "Hongkong", "Japan", "Pcso", "Singapore", "Sydney", "Taiwan"]

if 'scan_outputs' not in st.session_state: st.session_state.scan_outputs = {}
for label in DIGIT_LABELS:
    if f"win_{label}" not in st.session_state: st.session_state[f"win_{label}"] = 7
if "angka_list" not in st.session_state: st.session_state.angka_list = []

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Hari", ["harian", "kemarin", "2hari", "3hari"])
    putaran = st.number_input("🔁 Putaran", 10, 1000, 100)
    st.markdown("---")
    st.markdown("### 🎯 Opsi Prediksi")
    jumlah_digit = st.slider("🔢 Jumlah Digit Prediksi", 1, 9, 7) # Default ke 7 sesuai screenshot
    metode = st.selectbox("🧠 Metode", ["Markov", "LSTM AI", "Ensemble AI + Markov"])
    use_transformer = st.checkbox("🤖 Gunakan Transformer", value=True)
    model_type = "transformer" if use_transformer else "lstm"
    st.markdown("---")
    st.markdown("### 🪟 Window Size per Digit")
    window_per_digit = {}
    for label in DIGIT_LABELS:
        window_per_digit[label] = st.slider(f"{label.upper()}", 3, 30, st.session_state[f"win_{label}"], key=f"win_{label}")
    st.markdown("---")
    st.subheader("🔴 Live Result")
    live_results_data = fetch_live_results(lokasi_list)
    if live_results_data:
        df_live = pd.DataFrame(live_results_data)
        st.table(df_live)

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Ambil Data dari API", use_container_width=True):
        # ... logika ambil data
        pass
with col2:
    st.caption("Data angka akan digunakan untuk pelatihan dan prediksi.")
with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    riwayat_input = "\n".join(st.session_state.get("angka_list", []))
    riwayat_text = st.text_area("1 angka per baris:", riwayat_input, height=250, key="manual_input")
    if riwayat_text != riwayat_input:
        st.session_state.angka_list = [x.strip() for x in riwayat_text.splitlines() if x.strip().isdigit() and len(x.strip()) == 4]
        st.rerun()
df = pd.DataFrame({"angka": st.session_state.get("angka_list", [])})

tab_prediksi, tab_scan, tab_manajemen = st.tabs(["🔮 Prediksi & Hasil", "🪟 Scan Window Size", "⚙️ Manajemen Model"])

with tab_prediksi:
    if st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary"):
        max_ws_needed = max(window_per_digit.values())
        if len(df) < max_ws_needed + 1:
            st.warning(f"❌ Data tidak cukup. Butuh minimal {max_ws_needed + 1} baris.")
        else:
            result, _ = None, None
            with st.spinner("⏳ Memproses prediksi..."):
                if metode == "Markov": result, _ = top6_markov(df, top_n=jumlah_digit)
                # Tambahkan metode AI jika diperlukan
                # elif metode == "LSTM AI": result, _ = top_n_model(...)
            
            if result:
                st.subheader(f"🎯 Hasil Prediksi Top {jumlah_digit}")
                for i, label in enumerate(DIGIT_LABELS):
                    st.markdown(f"**{label.upper()}:** {', '.join(map(str, result[i]))}")

                st.divider()
                st.subheader("🎲 Acak 4D dari Hasil Prediksi (Sistem Rotasi)")
                
                if all(result) and len(result) == 4:
                    ribuan_list, ratusan_list, puluhan_list, satuan_list = result[0], result[1], result[2], result[3]
                    
                    patterns = [
                        (ribuan_list, ratusan_list, puluhan_list, satuan_list),
                        (ratusan_list, puluhan_list, satuan_list, ribuan_list),
                        (puluhan_list, satuan_list, ribuan_list, ratusan_list),
                        (satuan_list, ribuan_list, ratusan_list, puluhan_list)
                    ]

                    acak_4d_list = []
                    for _ in range(1000):
                        chosen_pattern = random.choice(patterns)
                        d1 = random.choice(chosen_pattern[0])
                        d2 = random.choice(chosen_pattern[1])
                        d3 = random.choice(chosen_pattern[2])
                        d4 = random.choice(chosen_pattern[3])
                        acak_4d_list.append(f"{d1}{d2}{d3}{d4}")
                    
                    output_string = " * ".join(acak_4d_list)
                    st.text_area(f"1000 Kombinasi Acak (Pola Rotasi)", output_string, height=300)
                else:
                    st.warning("Tidak bisa menghasilkan angka acak karena hasil prediksi tidak lengkap.")

with tab_manajemen:
    st.subheader("Manajemen Model AI")
    # ... Logika tab manajemen ...

with tab_scan:
    st.subheader("Pencarian Window Size (WS) Optimal per Digit")
    # ... Logika tab scan ...
