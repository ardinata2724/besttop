import streamlit as st
import pandas as pd
import os
import time
import random
import numpy as np
from collections import defaultdict, Counter
from itertools import product

# --- FUNGSI HELPER UNTUK MEMUAT MODEL (AMAN DIGUNAKAN) ---
@st.cache_resource
def _get_positional_encoding_layer():
    import tensorflow as tf
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
    return PositionalEncoding

@st.cache_resource
def load_cached_model(model_path):
    from tensorflow.keras.models import load_model
    PositionalEncoding = _get_positional_encoding_layer()
    if os.path.exists(model_path):
        try:
            return load_model(model_path, custom_objects={"PositionalEncoding": PositionalEncoding})
        except Exception as e:
            st.error(f"Gagal memuat model di {model_path}: {e}")
    return None

# --- FUNGSI-FUNGSI PREDIKSI ---
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
    unique_hasil = [list(dict.fromkeys(h))[:top_n] for h in hasil]
    return unique_hasil, None

def tf_preprocess_data(df, window_size=7):
    from tensorflow.keras.utils import to_categorical
    if len(df) < window_size + 1: return np.array([])
    angka = df["angka"].values
    sequences = []
    for i in range(len(angka) - window_size):
        window = [str(x).zfill(4) for x in angka[i:i+window_size]]
        sequences.append([int(d) for num in window for d in num])
    return np.array(sequences)

def top_n_model(df, lokasi, window_dict, model_type, top_n):
    results = []
    loc_id = lokasi.lower().strip().replace(" ", "_")
    DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]
    for label in DIGIT_LABELS:
        ws = window_dict.get(label, 7)
        X = tf_preprocess_data(df, window_size=ws)
        if X.shape[0] == 0: return None, None
        model_path = f"saved_models/{loc_id}_{label}_{model_type}.h5"
        model = load_cached_model(model_path)
        if model is None: 
            st.error(f"Model untuk {label} tidak ditemukan di {model_path}. Silakan latih dan unggah model terlebih dahulu.")
            return None, None
        
        pred = model.predict(X, verbose=0)
        avg = np.mean(pred, axis=0)
        results.append(list(avg.argsort()[-top_n:][::-1]))
    return results, None


# ==============================================================================
# APLIKASI STREAMLIT UTAMA
# ==============================================================================
st.set_page_config(page_title="Prediksi 4D", layout="wide")

if 'angka_list' not in st.session_state: st.session_state.angka_list = []

st.title("Prediksi 4D")
st.caption("editing by: Andi Prediction")

try: from lokasi_list import lokasi_list
except ImportError: lokasi_list = ["BULLSEYE", "HONGKONGPOOLS", "HONGKONG LOTTO", "SYDNEYPOOLS", "SYDNEY LOTTO", "SINGAPURA"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    putaran = st.number_input("🔁 Jumlah Putaran Terakhir", 10, 1000, 100)
    st.markdown("---")
    st.markdown("### 🎯 Opsi Prediksi")
    jumlah_digit = st.slider("🔢 Jumlah Digit Prediksi", 1, 9, 9)
    metode = st.selectbox("🧠 Metode", ["Markov", "LSTM AI"])
    use_transformer = st.checkbox("🤖 Gunakan Transformer", value=True)
    model_type = "transformer" if use_transformer else "lstm"
    st.markdown("---")
    st.markdown("### 🪟 Window Size per Digit (untuk Prediksi)")
    window_per_digit = {label: st.number_input(f"{label.upper()}", 1, 100, 7, key=f"win_{label}") for label in ["ribuan", "ratusan", "puluhan", "satuan"]}


def get_file_name_from_lokasi(lokasi):
    cleaned_lokasi = lokasi.lower().replace(" ", "")
    if "hongkonglotto" in cleaned_lokasi: return "keluaran hongkong lotto.txt"
    if "hongkongpools" in cleaned_lokasi: return "keluaran hongkongpools.txt"
    if "sydneylotto" in cleaned_lokasi: return "keluaran sydney lotto.txt"
    if "sydneypools" in cleaned_lokasi: return "keluaran sydneypools.txt"
    return f"keluaran {lokasi.lower()}.txt"

if st.button("Ambil Data dari Keluaran Angka", use_container_width=True):
    file_name = get_file_name_from_lokasi(selected_lokasi)
    try:
        with open(file_name, 'r') as f: lines = f.readlines()
        last_n_lines = lines[-putaran:]
        angka_from_file = [line.strip()[:4] for line in last_n_lines if line.strip() and line.strip()[:4].isdigit()]
        if angka_from_file:
            st.session_state.angka_list = angka_from_file
            st.success(f"{len(angka_from_file)} data berhasil diambil.")
    except FileNotFoundError: st.error(f"File tidak ditemukan: '{file_name}'.")
    except Exception as e: st.error(f"Terjadi kesalahan: {e}")

with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    riwayat_input = "\n".join(st.session_state.get("angka_list", []))
    riwayat_text = st.text_area("1 angka per baris:", riwayat_input, height=300, key="manual_data_input")
    if riwayat_text != riwayat_input:
        st.session_state.angka_list = [line.strip()[:4] for line in riwayat_text.splitlines() if line.strip() and line.strip()[:4].isdigit()]
        st.rerun()

df = pd.DataFrame({"angka": st.session_state.get("angka_list", [])})

st.divider()

if st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary"):
    if not df.empty and len(df) >= max(window_per_digit.values()) + 1:
        result, _ = None, None
        if metode == "Markov": result, _ = top6_markov(df, jumlah_digit)
        elif metode == "LSTM AI": result, _ = top_n_model(df, selected_lokasi, window_per_digit, model_type, jumlah_digit)
        
        if result:
            st.subheader(f"🎯 Hasil Prediksi Top {jumlah_digit}")
            DIGIT_LABELS_PRED = ["ribuan", "ratusan", "puluhan", "satuan"]
            for i, label in enumerate(DIGIT_LABELS_PRED): st.markdown(f"**{label.upper()}:** {', '.join(map(str, result[i]))}")
            st.divider()
            all_combinations = list(product(*result))
            st.subheader(f"🔢 Semua Kombinasi 4D ({len(all_combinations)} Line)")
            if len(all_combinations) > 5000: st.warning("Jumlah kombinasi sangat besar.")
            st.text_area("Kombinasi Penuh", " * ".join(["".join(map(str, combo)) for combo in all_combinations]), height=300)
    else:
        st.warning("❌ Data tidak cukup untuk prediksi.")
