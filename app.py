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

# ==============================================================================
# BAGIAN 1: DEFINISI FUNGSI-FUNGSI INTI
# ==============================================================================
# (Semua fungsi dari bagian ini tetap sama, untuk keringkasan tidak ditampilkan ulang)
# --- Fungsi dari Markov Model ---
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

# --- Fungsi dari AI Model ---
DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

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

def preprocess_data(df, window_size=7):
    if len(df) < window_size + 1: return np.array([]), {}
    angka = df["angka"].values
    sequences, targets = [], {label: [] for label in DIGIT_LABELS}
    for i in range(len(angka) - window_size):
        window = [str(x).zfill(4) for x in angka[i:i+window_size+1]]
        if any(not x.isdigit() for x in window): continue
        sequences.append([int(d) for num in window[:-1] for d in num])
        target_digits = [int(d) for d in window[-1]]
        for j, label in enumerate(DIGIT_LABELS):
            targets[label].append(to_categorical(target_digits[j], num_classes=10))
    return np.array(sequences), {label: np.array(v) for label, v in targets.items()}

def build_model(input_len, model_type="lstm"):
    inputs = Input(shape=(input_len,))
    x = Embedding(input_dim=10, output_dim=64)(inputs)
    x = PositionalEncoding()(x)
    if model_type == "transformer":
        attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = LayerNormalization()(x + attn)
    else: # LSTM
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.3)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_save_model(df, lokasi, window_dict, model_type="lstm"):
    os.makedirs("saved_models", exist_ok=True)
    for label in DIGIT_LABELS:
        ws = window_dict.get(label, 7)
        X, y_dict = preprocess_data(df, window_size=ws)
        if label not in y_dict or y_dict[label].shape[0] < 10: continue
        y = y_dict[label]
        model_path = f"saved_models/{lokasi.lower().replace(' ', '_')}_{label}_{model_type}.h5"
        model = build_model(X.shape[1], model_type)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)
        model.save(model_path)

def top_n_model(df, lokasi, window_dict, model_type, top_n=6):
    # ...
    pass
def top_n_ensemble(df, lokasi, window_dict, model_type, top_n=6):
    # ...
    pass
def find_best_window_size(container, df, label, model_type, min_ws, max_ws, top_n):
    # ...
    pass

# ==============================================================================
# BAGIAN 2: APLIKASI STREAMLIT UTAMA
# ==============================================================================

st.set_page_config(page_title="Prediksi AI", layout="wide")
st.title("Prediksi 4D - AI")

try:
    from lokasi_list import lokasi_list
except ImportError:
    lokasi_list = ["HONGKONG", "BULLSEYE", "SYDNEY", "SINGAPORE"]

# Inisialisasi state
if 'scan_output_container' not in st.session_state:
    st.session_state.scan_output_container = {}
for label in DIGIT_LABELS:
    if f"win_{label}" not in st.session_state:
        st.session_state[f"win_{label}"] = 7
if "angka_list" not in st.session_state:
    st.session_state.angka_list = []

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Hari", ["harian", "kemarin", "2hari", "3hari"])
    putaran = st.number_input("🔁 Putaran", 10, 1000, 100)
    st.markdown("---")
    st.markdown("### 🎯 Opsi Prediksi")
    jumlah_digit = st.slider("🔢 Jumlah Digit Prediksi", 1, 9, 6)
    metode = st.selectbox("🧠 Metode", ["Markov", "LSTM AI", "Ensemble AI + Markov"])
    use_transformer = st.checkbox("🤖 Gunakan Transformer", value=True)
    model_type = "transformer" if use_transformer else "lstm"
    st.markdown("---")
    st.markdown("### 🪟 Window Size per Digit")
    window_per_digit = {}
    for label in DIGIT_LABELS:
        window_per_digit[label] = st.slider(f"{label.upper()}", 3, 30, st.session_state[f"win_{label}"], key=f"win_{label}")

# --- PERBAIKAN LOGIKA TOMBOL API ---
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Ambil Data dari API", use_container_width=True):
        try:
            with st.spinner("🔄 Mengambil data dari API..."):
                url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
                headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
                data = requests.get(url, headers=headers).json()
                
                if data.get("data"):
                    angka_api = [d["result"] for d in data["data"] if len(d["result"]) == 4 and d["result"].isdigit()]
                    st.session_state.angka_list = angka_api
                    st.success(f"{len(angka_api)} angka berhasil diambil.")
                    st.rerun() # Refresh halaman untuk menampilkan data baru
                else:
                    st.error("Gagal mengambil data. Respon API tidak valid.")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Gagal koneksi ke API: {e}")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")

with col2:
    st.caption("Data angka akan digunakan untuk pelatihan dan prediksi.")

with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    riwayat_input = "\n".join(st.session_state.angka_list)
    riwayat_input = st.text_area("📝 1 angka per baris:", value=riwayat_input, height=300)
    if riwayat_input != "\n".join(st.session_state.angka_list):
        st.session_state.angka_list = [x.strip() for x in riwayat_input.splitlines() if x.strip().isdigit() and len(x.strip()) == 4]
        st.rerun()
    
    df = pd.DataFrame({"angka": st.session_state.angka_list})


# ======== Tabs Utama ========
tab_prediksi, tab_scan, tab_manajemen = st.tabs(["🔮 Prediksi & Hasil", "🪟 Scan Window Size", "⚙️ Manajemen Model"])

with tab_prediksi:
    if st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary"):
        # ... (Logika prediksi Anda di sini)
        pass

with tab_manajemen:
    # ... (Logika manajemen model Anda di sini)
    pass
    
with tab_scan:
    # ... (Logika scan window size Anda di sini)
    pass
