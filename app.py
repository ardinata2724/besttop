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
# ==============================================================================

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

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

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
    results, probs = [], []
    loc_id = lokasi.lower().replace(" ", "_")
    for label in DIGIT_LABELS:
        ws = window_dict.get(label, 7)
        X, _ = preprocess_data(df, window_size=ws)
        if X.shape[0] == 0: return None, None
        model_path = f"saved_models/{loc_id}_{label}_{model_type}.h5"
        if not os.path.exists(model_path): return None, None
        try:
            model = load_model(model_path, custom_objects={"PositionalEncoding": PositionalEncoding})
            pred = model.predict(X, verbose=0)
            avg = np.mean(pred, axis=0)
            top_indices = avg.argsort()[-top_n:][::-1]
            results.append(list(top_indices))
            probs.append([avg[i] for i in top_indices])
        except Exception as e:
            st.error(f"Error memuat model untuk {label}: {e}"); return None, None
    return results, probs

# ... (sisa fungsi inti tidak berubah)

# ==============================================================================
# BAGIAN 2: APLIKASI STREAMLIT UTAMA
# ==============================================================================

st.set_page_config(page_title="Prediksi AI", layout="wide")
st.title("Prediksi 4D - AI")

try: from lokasi_list import lokasi_list
except ImportError: lokasi_list = ["HONGKONG", "BULLSEYE", "SYDNEY", "SINGAPORE"]

# Inisialisasi state
# ... (tidak berubah)

with st.sidebar:
    # ... (sidebar tidak berubah)
    pass

# ... (UI Data Loader tidak berubah)
df = pd.DataFrame() # Placeholder

tab_prediksi, tab_scan, tab_manajemen = st.tabs(["🔮 Prediksi & Hasil", "🪟 Scan Window Size", "⚙️ Manajemen Model"])

with tab_prediksi:
    if st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary"):
        # ... (logika prediksi tidak berubah)
        
        # --- PERUBAHAN DI BLOK INI ---
        if result:
            st.subheader(f"🎯 Hasil Prediksi Top {jumlah_digit}")
            # ... (kode tampilan hasil prediksi tidak berubah)
            
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
                # --- PERUBAHAN UTAMA: DARI 1000 MENJADI 2000 ---
                for _ in range(2000):
                    chosen_pattern = random.choice(patterns)
                    d1 = random.choice(chosen_pattern[0])
                    d2 = random.choice(chosen_pattern[1])
                    d3 = random.choice(chosen_pattern[2])
                    d4 = random.choice(chosen_pattern[3])
                    acak_4d_list.append(f"{d1}{d2}{d3}{d4}")
                
                output_string = " * ".join(acak_4d_list)
                # --- PERUBAHAN JUDUL ---
                st.text_area(f"2000 Kombinasi Acak (Pola Rotasi)", output_string, height=300)

with tab_manajemen:
    # ... (tidak berubah)
    pass

with tab_scan:
    # ... (tidak berubah)
    pass
