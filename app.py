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
# BAGIAN 1: DEFINISI SEMUA FUNGSI-FUNGSI INTI
# ==============================================================================
DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

def _ensure_unique_top_n(top_list, n=6):
    """Memastikan daftar top-N memiliki item unik hingga N."""
    unique_list = list(dict.fromkeys(top_list))[:n]
    if len(unique_list) >= n: return unique_list
    all_digits = list(range(10)); random.shuffle(all_digits)
    unique_set = set(unique_list)
    for digit in all_digits:
        if len(unique_set) >= n: break
        if digit not in unique_set: unique_set.add(digit)
    return list(unique_set)

def top6_markov(df, top_n=6):
    """Prediksi menggunakan metode Markov Chain sederhana."""
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

def calculate_angka_main(df, top_n=5):
    """
    Menghitung berbagai jenis 'Angka Main' dari data historis.
    Jumlah hasil ditentukan oleh parameter top_n.
    """
    if df.empty or len(df) < 10:
        return {
            "ai_depan": "Data tidak cukup",
            "ai_tengah": "Data tidak cukup",
            "ai_belakang": "Data tidak cukup",
            "jumlah_2d": "Data tidak cukup",
            "colok_bebas": "Data tidak cukup",
            "ai_3d": "Data tidak cukup",
        }

    angka_str = df["angka"].astype(str).str.zfill(4)
    
    # AI 2D (Depan, Tengah, Belakang) - diubah menjadi multiline
    depan = angka_str.str[:2]
    ai_depan = "\n".join(depan.value_counts().nlargest(top_n).index)
    
    tengah = angka_str.str[1:3]
    ai_tengah = "\n".join(tengah.value_counts().nlargest(top_n).index)
    
    belakang = angka_str.str[2:]
    ai_belakang = "\n".join(belakang.value_counts().nlargest(top_n).index)
    
    # Jumlah 2D (berdasarkan digit belakang)
    puluhan = angka_str.str[2].astype(int)
    satuan = angka_str.str[3].astype(int)
    jumlah = (puluhan + satuan) % 10
    jumlah_2d = ", ".join(map(str, jumlah.value_counts().nlargest(top_n).index))
    
    # Colok Bebas (digit paling sering muncul)
    all_digits = "".join(angka_str.tolist())
    colok_bebas = ", ".join([item[0] for item in Counter(all_digits).most_common(top_n)])
    
    # AI 3D (berdasarkan 3 digit belakang) - diubah menjadi multiline
    ai_3d_series = angka_str.str[1:]
    ai_3d = "\n".join(ai_3d_series.value_counts().nlargest(top_n).index)

    return {
        "ai_depan": ai_depan,
        "ai_tengah": ai_tengah,
        "ai_belakang": ai_belakang,
        "jumlah_2d": jumlah_2d,
        "colok_bebas": colok_bebas,
        "ai_3d": ai_3d,
    }

class PositionalEncoding(tf.keras.layers.Layer):
    """Layer untuk menambahkan positional encoding pada model Transformer."""
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
    """Mempersiapkan data sekuensial untuk model AI."""
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
    """Membangun arsitektur model AI (LSTM atau Transformer)."""
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

def top_n_model(df, lokasi, window_dict, model_type, top_n=6):
    """Melakukan prediksi menggunakan model AI yang sudah dilatih."""
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

def find_best_window_size(df, label, model_type, min_ws, max_ws, top_n):
    """Mencari window size terbaik dengan melatih dan mengevaluasi model."""
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
            preds = model.predict(X_val, verbose=0)
            avg_conf = np.mean(np.sort(preds, axis=1)[:, -top_n:]) * 100
            last_pred = model.predict(X[-1:], verbose=0)[0]
            top_n_digits_pred = ", ".join(map(str, np.argsort(last_pred)[::-1][:top_n]))
            score = (acc * 0.2) + (top_n_acc * 0.5) + (avg_conf/100 * 0.3)
            table_data.append((ws, f"{acc*100:.2f}", f"{top_n_acc*100:.2f}", f"{avg_conf:.2f}", top_n_digits_pred))
            if score > best_score: best_score, best_ws = score, ws
        except Exception: continue
    progress_bar.empty()
    if not table_data: return None, None
    return best_ws, pd.DataFrame(table_data, columns=["Window Size", "Acc (%)", f"Top-{top_n} Acc (%)", "Conf (%)", f"Top-{top_n}"])

def train_and_save_model(df, lokasi, window_dict, model_type):
    """Melatih dan menyimpan model untuk setiap posisi digit."""
    st.info(f"Memulai proses pelatihan untuk lokasi: {lokasi} (Model: {model_type.upper()})")
    lokasi_id = lokasi.lower().strip().replace(" ", "_")
    
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
        st.toast("Direktori 'saved_models' dibuat.")

    for label in DIGIT_LABELS:
        ws = window_dict.get(label, 7)
        progress_text = f"Memproses data untuk digit {label.upper()} dengan Window Size = {ws}..."
        bar = st.progress(0, text=progress_text)

        X, y_dict = preprocess_data(df, window_size=ws)
        bar.progress(25, text=progress_text)

        if label not in y_dict or y_dict[label].shape[0] < 10:
            st.warning(f"Data tidak cukup untuk melatih model '{label.upper()}'. Minimal butuh 10 data. Dilewati.")
            bar.empty()
            continue
        
        y = y_dict[label]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        progress_text = f"Melatih model untuk {label.upper()}... Ini mungkin akan memakan waktu."
        bar.progress(50, text=progress_text)

        model = build_model(X.shape[1], model_type)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
        
        bar.progress(75, text=f"Menyimpan model untuk {label.upper()}...")
        model_path = f"saved_models/{lokasi_id}_{label}_{model_type}.h5"
        model.save(model_path)
        
        bar.progress(100, text=f"Model untuk {label.upper()} berhasil dilatih & disimpan!")
        time.sleep(1)
        bar.empty()

# ==============================================================================
# BAGIAN 2: APLIKASI STREAMLIT UTAMA
# ==============================================================================
st.set_page_config(page_title="Prediksi AI", layout="wide")
st.title("Prediksi 4D - AI")

try: from lokasi_list import lokasi_list
except ImportError: lokasi_list = ["BULLSEYE", "HONGKONG", "SYDNEY", "SINGAPORE"]

# Inisialisasi session state
if 'scan_outputs' not in st.session_state: st.session_state.scan_outputs = {}
for label in DIGIT_LABELS:
    if f"win_{label}" not in st.session_state: st.session_state[f"win_{label}"] = 7
if "angka_list" not in st.session_state: st.session_state.angka_list = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    selected_hari = st.selectbox("📅 Hari", ["harian", "kemarin", "2hari", "3hari"])
    putaran = st.number_input("🔁 Putaran", 10, 1000, 100)
    st.markdown("---")
    st.markdown("### 🎯 Opsi Prediksi")
    jumlah_digit = st.slider("🔢 Jumlah Digit Prediksi", 1, 9, 4) # Mengubah default ke 4
    metode = st.selectbox("🧠 Metode", ["Markov", "LSTM AI"])
    use_transformer = st.checkbox("🤖 Gunakan Transformer", value=True)
    model_type = "transformer" if use_transformer else "lstm"
    st.markdown("---")
    st.markdown("### 🪟 Window Size per Digit")
    window_per_digit = {}
    for label in DIGIT_LABELS:
        window_per_digit[label] = st.number_input(
            f"{label.upper()}", 
            min_value=1, 
            max_value=100, 
            value=st.session_state[f"win_{label}"], 
            key=f"win_{label}"
        )

# --- KONTEN UTAMA ---
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Ambil Data dari API", use_container_width=True):
        try:
            with st.spinner("🔄 Mengambil data..."):
                url = f"https://wysiwygscan.com/api?pasaran={selected_lokasi.lower()}&hari={selected_hari}&putaran={putaran}&format=json&urut=asc"
                headers = {"Authorization": "Bearer 6705327a2c9a9135f2c8fbad19f09b46"}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                if data.get("data"):
                    angka_api = [d["result"] for d in data["data"] if len(str(d.get("result", ""))) == 4 and str(d.get("result", "")).isdigit()]
                    st.session_state.angka_list = angka_api
                    st.success(f"{len(angka_api)} angka berhasil diambil.")
                else:
                    st.error("API tidak mengembalikan data yang valid.")
        except Exception as e:
            st.error(f"Gagal mengambil data dari API: {e}")

with col2: st.caption("Data angka akan digunakan untuk pelatihan dan prediksi.")
with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    riwayat_input = "\n".join(st.session_state.get("angka_list", []))
    riwayat_text = st.text_area("1 angka per baris:", riwayat_input, height=250)
    if riwayat_text != riwayat_input:
        st.session_state.angka_list = [x.strip() for x in riwayat_text.splitlines() if x.strip().isdigit() and len(x.strip()) == 4]
        st.rerun()
df = pd.DataFrame({"angka": st.session_state.get("angka_list", [])})

# --- Definisi Tab ---
tab_prediksi, tab_scan, tab_manajemen, tab_angka_main = st.tabs([
    "🔮 Prediksi & Hasil", 
    "🪟 Scan Window Size", 
    "⚙️ Manajemen Model",
    "🎯 Angka Main"
])

# --- TAB PREDIKSI ---
with tab_prediksi:
    if st.button("🚀 Jalankan Prediksi", use_container_width=True, type="primary"):
        max_ws = max(window_per_digit.values())
        if len(df) < max_ws + 1:
            st.warning(f"❌ Data tidak cukup. Butuh minimal {max_ws + 1} baris.")
        else:
            result, _ = None, None
            with st.spinner("⏳ Memproses prediksi..."):
                if metode == "Markov":
                    result, _ = top6_markov(df, top_n=jumlah_digit)
                elif metode == "LSTM AI":
                    result, _ = top_n_model(df, selected_lokasi, window_per_digit, model_type, jumlah_digit)
            
            if result and all(result):
                st.subheader(f"🎯 Hasil Prediksi Top {jumlah_digit}")
                for i, label in enumerate(DIGIT_LABELS):
                    st.markdown(f"**{label.upper()}:** {', '.join(map(str, result[i]))}")
                
                st.divider()
                all_combinations = list(product(*result))
                kombinasi_4d_list = ["".join(map(str, combo)) for combo in all_combinations]
                total_kombinasi = len(kombinasi_4d_list)
                st.subheader(f"🔢 Semua Kombinasi 4D ({total_kombinasi} Line)")
                output_string = " * ".join(kombinasi_4d_list)
                st.text_area(f"Total {total_kombinasi} Kombinasi Penuh", output_string, height=300)

# --- TAB MANAJEMEN MODEL ---
with tab_manajemen:
    st.subheader("Manajemen Model AI")
    st.info("Latih atau hapus model AI di sini. Jika Anda mengubah Window Size, latih ulang model untuk hasil terbaik.")
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
    if st.button("📚 Latih & Simpan Semua Model AI", use_container_width=True, type="primary"):
        max_ws = max(window_per_digit.values())
        if len(df) < max_ws + 10:
            st.error(f"Data tidak cukup untuk melatih. Butuh setidaknya {max_ws + 10} baris.")
        else:
            train_and_save_model(df, selected_lokasi, window_per_digit, model_type)
            st.success("✅ Semua model berhasil dilatih!"); st.rerun()

# --- TAB SCAN WINDOW SIZE ---
with tab_scan:
    st.subheader("Pencarian Window Size (WS) Optimal per Digit")
    st.info("Klik tombol scan untuk setiap digit. Hasilnya akan muncul dan tetap ada. Setelah menemukan WS terbaik, **atur slider di sidebar secara manual**.")
    scan_cols = st.columns(2)
    min_ws = scan_cols[0].number_input("Min WS", 1, 99, 3)
    max_ws = scan_cols[1].number_input("Max WS", min_ws + 1, 100, 25)
    
    if st.button("❌ Hapus Hasil Scan"):
        st.session_state.scan_outputs = {}
        st.rerun()
    st.divider()

    btn_cols = st.columns(4)
    for i, label in enumerate(DIGIT_LABELS):
        if btn_cols[i].button(f"🔎 Scan {label.upper()}", use_container_width=True):
            st.toast(f"🔎 Sedang memindai {label.upper()}, mohon tunggu...", icon="⏳")
            st.session_state.scan_outputs[label] = "PENDING"
            st.rerun()

    for label in [l for l in DIGIT_LABELS if l in st.session_state.scan_outputs]:
        data = st.session_state.scan_outputs[label]
        with st.expander(f"Hasil Scan untuk {label.upper()}", expanded=True):
            if data == "PENDING":
                best_ws, result_table = find_best_window_size(df, label, model_type, min_ws, max_ws, jumlah_digit)
                st.session_state.scan_outputs[label] = {"ws": best_ws, "table": result_table}
                st.rerun()
            
            elif isinstance(data, dict):
                result_df = data.get("table")
                if result_df is not None and not result_df.empty:
                    st.dataframe(result_df)
                    st.markdown("---")
                    st.markdown("👇 **Salin Hasil dari Kolom Top-N**")
                    
                    top_n_column_name = f"Top-{jumlah_digit}"

                    if top_n_column_name in result_df.columns:
                        copyable_text = "\n".join(result_df[top_n_column_name].astype(str))
                        st.text_area(
                            "Klik di dalam kotak di bawah lalu tekan Ctrl+A dan Ctrl+C untuk menyalin semua baris.",
                            value=copyable_text,
                            height=250
                        )

                    if data["ws"] is not None:
                        st.success(f"✅ WS terbaik yang disarankan: {data['ws']}")
                    else:
                        st.warning("Tidak ditemukan WS yang menonjol.")
                else:
                    st.warning("Tidak ada hasil yang ditemukan.")

# --- TAB ANGKA MAIN ---
with tab_angka_main:
    st.subheader("Analisis Angka Main dari Data Historis")
    
    if len(df) < 10:
        st.warning("Data historis tidak cukup untuk melakukan analisis (minimal 10 baris).")
    else:
        with st.spinner("Menganalisis Angka Main..."):
            # Memanggil fungsi dengan jumlah digit dari sidebar
            am = calculate_angka_main(df, top_n=jumlah_digit)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Analisis 2D & 3D")
                # Menggunakan text_area untuk menampilkan hasil multiline
                st.markdown("**AI Depan**")
                st.text_area("ai_depan_area", am['ai_depan'], height=(jumlah_digit * 20) + 10, disabled=True, label_visibility="collapsed")
                
                st.markdown("**AI Tengah**")
                st.text_area("ai_tengah_area", am['ai_tengah'], height=(jumlah_digit * 20) + 10, disabled=True, label_visibility="collapsed")
                
                st.markdown("**AI Belakang**")
                st.text_area("ai_belakang_area", am['ai_belakang'], height=(jumlah_digit * 20) + 10, disabled=True, label_visibility="collapsed")

            with col2:
                st.markdown("##### Analisis Lainnya")
                st.markdown(f"**Jumlah 2D (Belakang):** `{am['jumlah_2d']}`")
                st.markdown(f"**Colok Bebas:** `{am['colok_bebas']}`")
                
                st.markdown("**AI 3D (Belakang)**")
                st.text_area("ai_3d_area", am['ai_3d'], height=(jumlah_digit * 20) + 10, disabled=True, label_visibility="collapsed")

    st.info("Angka di atas adalah hasil analisis statistik dari data historis yang tersedia dan bukan merupakan jaminan hasil.")
