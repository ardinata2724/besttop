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
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# BAGIAN 1: DEFINISI FUNGSI-FUNGSI INTI
# ==============================================================================

# (Fungsi-fungsi dari Markov dan AI Model diletakkan di sini, sama seperti sebelumnya)
# ... (Untuk keringkasan, saya tidak menampilkan ulang semua fungsi, tapi pastikan Anda menyalin seluruh blok kode ini)
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
    else:
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
    results = []
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
        except Exception as e:
            st.error(f"Error memuat model untuk {label}: {e}"); return None, None
    return results, None

def top_n_ensemble(df, lokasi, window_dict, model_type, top_n=6):
    ai_result, _ = top_n_model(df, lokasi, window_dict, model_type, top_n)
    markov_result, _ = top6_markov(df, top_n)
    if ai_result is None or markov_result is None: return None
    ensemble = []
    for i in range(4):
        combined = list(dict.fromkeys(ai_result[i] + markov_result[i]))
        ensemble.append(combined[:top_n])
    return ensemble, None

def find_best_window_size(container, df, label, model_type, min_ws, max_ws, top_n):
    best_ws, best_score = None, -1
    table_data = []
    for ws in range(min_ws, max_ws + 1):
        try:
            X, y_dict = preprocess_data(df, window_size=ws)
            if label not in y_dict or y_dict[label].shape[0] < 10: continue
            y = y_dict[label]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model = build_model(X.shape[1], model_type)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", TopKCategoricalAccuracy(k=top_n)])
            model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=0)
            _, acc, top_n_acc = model.evaluate(X_val, y_val, verbose=0)
            score = (acc * 0.4) + (top_n_acc * 0.6)
            table_data.append((ws, f"{acc:.2%}", f"{top_n_acc:.2%}", f"{score:.2f}"))
            if score > best_score:
                best_score, best_ws = score, ws
        except Exception: continue
    
    with container:
        if not table_data:
            st.error("Tidak ada data yang cukup untuk di-scan pada rentang WS ini.")
            return
        
        df_table = pd.DataFrame(table_data, columns=["Window Size", "Akurasi Top-1", f"Akurasi Top-{top_n}", "Skor"])
        st.dataframe(df_table)
        if best_ws is not None:
            st.success(f"✅ WS terbaik untuk {label.upper()}: {best_ws}")
        else:
            st.warning(f"Tidak ditemukan WS yang memenuhi kriteria untuk {label.upper()}.")

# ==============================================================================
# BAGIAN 2: APLIKASI STREAMLIT UTAMA
# ==============================================================================

# --- BARIS PENTING UNTUK MEMBUAT LAYOUT LEBAR ---
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

# --- UI Sidebar ---
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

# --- UI Data Loader ---
if "angka_list" not in st.session_state: st.session_state.angka_list = []
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Ambil Data dari API", use_container_width=True):
        # ... logika ambil data
        pass
with col2:
    st.caption("Data angka akan digunakan untuk pelatihan dan prediksi.")
with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    riwayat_input = "\n".join(st.session_state.get("angka_list", []))
    riwayat_input = st.text_area("📝 1 angka per baris:", value=riwayat_input, height=300)
    st.session_state.angka_list = [x.strip() for x in riwayat_input.splitlines() if x.strip().isdigit() and len(x.strip()) == 4]
    df = pd.DataFrame({"angka": st.session_state.get("angka_list", [])})

# ======== Tabs Utama ========
tab_prediksi, tab_scan, tab_manajemen = st.tabs(["🔮 Prediksi & Hasil", "🪟 Scan Window Size", "⚙️ Manajemen Model"])

with tab_prediksi:
    if st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary"):
        max_ws_needed = max(list(window_per_digit.values()))
        if len(df) < max_ws_needed + 1:
            st.warning(f"❌ Data tidak cukup. Butuh minimal {max_ws_needed + 1} baris data.")
        else:
            with st.spinner("⏳ Memproses prediksi..."):
                result, _ = None, None
                if metode == "Markov":
                    result, _ = top6_markov(df, top_n=jumlah_digit)
                elif metode == "LSTM AI":
                    result, _ = top_n_model(df, selected_lokasi, window_per_digit, model_type, jumlah_digit)
                    if result is None: st.error("Gagal memuat model AI. Pastikan model sudah dilatih.")
                elif metode == "Ensemble AI + Markov":
                    result, _ = top_n_ensemble(df, selected_lokasi, window_per_digit, model_type, jumlah_digit)
                    if result is None: st.error("Gagal prediksi ensemble. Pastikan model AI sudah dilatih.")
            
            if result:
                st.subheader(f"🎯 Hasil Prediksi Top {jumlah_digit}")
                for i, label in enumerate(DIGIT_LABELS):
                    st.markdown(f"**{label.upper()}:** {', '.join(map(str, result[i]))}")

with tab_manajemen:
    st.subheader("Manajemen Model AI")
    st.info("Latih atau hapus model AI di sini.")
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
        if len(df) < max_ws_needed + 10:
            st.error(f"Data tidak cukup untuk melatih. Butuh setidaknya {max_ws_needed + 10} baris data.")
        else:
            with st.spinner("🔄 Melatih semua model..."):
                train_and_save_model(df, selected_lokasi, window_per_digit, model_type=model_type)
            st.success("✅ Semua model berhasil dilatih!"); st.rerun()

with tab_scan:
    st.subheader("Pencarian Window Size (WS) Optimal per Digit")
    st.info("Klik tombol scan untuk setiap digit. Hasilnya akan muncul dan tetap ada di bawah. Setelah itu, atur slider di sidebar secara manual.")

    scan_cols = st.columns(2)
    min_ws = scan_cols[0].number_input("Min WS", 3, 20, 3)
    max_ws = scan_cols[1].number_input("Max WS", min_ws + 1, 30, 12)
    
    btn_cols = st.columns(4)
    for i, label in enumerate(DIGIT_LABELS):
        if btn_cols[i].button(f"🔎 Scan {label.upper()}", use_container_width=True):
            st.session_state.scan_output_container[label] = True

    if st.button("❌ Hapus Hasil Scan"):
        st.session_state.scan_output_container = {}
        st.rerun()
    
    st.divider()

    sorted_labels = [l for l in DIGIT_LABELS if st.session_state.scan_output_container.get(l)]
    for label in sorted_labels:
        container = st.expander(f"Hasil Scan untuk {label.upper()}", expanded=True)
        with container:
            find_best_window_size(st, df, label, model_type, min_ws, max_ws, jumlah_digit)
