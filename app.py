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

def top_n_ensemble(df, lokasi, window_dict, model_type, top_n=6):
    ai_result, _ = top_n_model(df, lokasi, window_dict, model_type, top_n)
    markov_result, _ = top6_markov(df, top_n)
    if ai_result is None or markov_result is None: return None, None
    ensemble = []
    for i in range(4):
        combined = list(dict.fromkeys(ai_result[i] + markov_result[i]))
        ensemble.append(combined[:top_n])
    return ensemble, None

# --- FUNGSI SCAN WINDOW SIZE DIPERBARUI UNTUK OUTPUT LENGKAP ---
def find_best_window_size(df, label, model_type, min_ws, max_ws, top_n):
    best_ws, best_score = None, -1
    table_data = []
    progress_bar = st.progress(0.0, text=f"Memulai scan untuk {label.upper()}...")
    total_steps = max_ws - min_ws + 1

    for i, ws in enumerate(range(min_ws, max_ws + 1)):
        progress_bar.progress((i + 1) / total_steps, text=f"Mencoba WS={ws} untuk {label.upper()}...")
        try:
            X, y_dict = preprocess_data(df, window_size=ws)
            if label not in y_dict or y_dict[label].shape[0] < 10: continue
            
            y = y_dict[label]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = build_model(X.shape[1], model_type)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", TopKCategoricalAccuracy(k=top_n)])
            model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=0)
            
            _, acc, top_n_acc = model.evaluate(X_val, y_val, verbose=0)
            
            # Hitung confidence dan ambil contoh Top-N
            preds = model.predict(X_val, verbose=0)
            avg_conf = np.mean(np.sort(preds, axis=1)[:, -top_n:]) * 100
            
            last_pred = model.predict(X[-1:], verbose=0)[0]
            top_n_digits_pred = ", ".join(map(str, np.argsort(last_pred)[::-1][:top_n]))
            
            # Skor internal untuk menentukan yang terbaik
            score = (acc * 0.2) + (top_n_acc * 0.5) + (avg_conf/100 * 0.3)
            
            table_data.append((ws, f"{acc*100:.2f}", f"{top_n_acc*100:.2f}", f"{avg_conf:.2f}", top_n_digits_pred))

            if score > best_score:
                best_score, best_ws = score, ws
        except Exception:
            continue
    
    progress_bar.empty()
    
    if not table_data:
        st.error("Tidak ada data yang cukup untuk di-scan pada rentang WS ini.")
        return None, None
        
    df_table = pd.DataFrame(table_data, columns=["Window Size", "Acc (%)", f"Top-{top_n} Acc (%)", "Conf (%)", f"Top-{top_n}"])
    return best_ws, df_table


# ==============================================================================
# BAGIAN 2: APLIKASI STREAMLIT UTAMA
# ==============================================================================

st.set_page_config(page_title="Prediksi AI", layout="wide")
st.title("Prediksi 4D - AI")

try: from lokasi_list import lokasi_list
except ImportError: lokasi_list = ["HONGKONG", "BULLSEYE", "SYDNEY", "SINGAPORE"]

if 'scan_outputs' not in st.session_state: st.session_state.scan_outputs = {}
for label in DIGIT_LABELS:
    if f"win_{label}" not in st.session_state: st.session_state[f"win_{label}"] = 7
if "angka_list" not in st.session_state: st.session_state.angka_list = []

with st.sidebar:
    # ... (Kode sidebar tidak berubah) ...
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

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Ambil Data dari API", use_container_width=True):
        # ... (Logika ambil data Anda) ...
        pass
with col2:
    st.caption("Data angka akan digunakan untuk pelatihan dan prediksi.")
with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    riwayat_input = "\n".join(st.session_state.get("angka_list", []))
    riwayat_input = st.text_area("📝 1 angka per baris:", value=riwayat_input, height=300, key="manual_input")
    if riwayat_input != "\n".join(st.session_state.angka_list):
        st.session_state.angka_list = [x.strip() for x in riwayat_input.splitlines() if x.strip().isdigit() and len(x.strip()) == 4]
        st.rerun()
df = pd.DataFrame({"angka": st.session_state.get("angka_list", [])})

# ======== Tabs Utama ========
tab_prediksi, tab_scan, tab_manajemen = st.tabs(["🔮 Prediksi & Hasil", "🪟 Scan Window Size", "⚙️ Manajemen Model"])

with tab_prediksi:
    if st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary"):
        # ... (Logika prediksi Anda) ...
        pass

with tab_manajemen:
    st.subheader("Manajemen Model AI")
    # ... (Logika manajemen Anda) ...
    pass

with tab_scan:
    st.subheader("Pencarian Window Size (WS) Optimal per Digit")
    st.info("Klik tombol scan untuk setiap digit. Hasilnya akan muncul dan tetap ada di bawah. Setelah menemukan WS terbaik, **atur slider di sidebar secara manual**.")
    scan_cols = st.columns(2)
    min_ws = scan_cols[0].number_input("Min WS", 3, 20, 3)
    max_ws = scan_cols[1].number_input("Max WS", min_ws + 1, 30, 12)
    
    btn_cols = st.columns(4)
    for i, label in enumerate(DIGIT_LABELS):
        if btn_cols[i].button(f"🔎 Scan {label.upper()}", use_container_width=True):
            if len(df) < max_ws + 5:
                st.error(f"Data tidak cukup. Butuh minimal {max_ws + 5} baris data.")
            else:
                # Tandai untuk diproses
                st.session_state.scan_outputs[label] = "PENDING" 
                st.rerun()

    if st.button("❌ Hapus Hasil Scan"):
        st.session_state.scan_outputs = {}
        st.rerun()
    st.divider()

    # Selalu tampilkan semua hasil yang tersimpan atau proses yang sedang berjalan
    for label, data in st.session_state.scan_outputs.items():
        with st.expander(f"Hasil Scan untuk {label.upper()}", expanded=True):
            if data == "PENDING":
                best_ws, result_table = find_best_window_size(df, label, model_type, min_ws, max_ws, jumlah_digit)
                # Simpan hasil permanen
                st.session_state.scan_outputs[label] = {"ws": best_ws, "table": result_table}
                st.rerun() # Refresh untuk menampilkan hasil
            
            elif isinstance(data, dict):
                if data.get("table") is not None and not data["table"].empty:
                    st.dataframe(data["table"])
                    if data["ws"] is not None:
                        st.success(f"✅ WS terbaik yang disarankan: {data['ws']}")
                    else:
                        st.warning("Tidak ada WS yang menonjol berdasarkan skor.")
                else:
                    st.warning("Tidak ada hasil yang ditemukan untuk kriteria ini.")
