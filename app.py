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
BBFS_LABELS = ["bbfs_ribuan-ratusan", "bbfs_ratusan-puluhan", "bbfs_puluhan-satuan"]
JUMLAH_LABELS = ["jumlah_depan", "jumlah_tengah", "jumlah_belakang"]
SHIO_LABELS = ["shio_depan", "shio_tengah", "shio_belakang"]


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

def calculate_angka_main_stats(df, top_n=5):
    """Menghitung statistik sederhana seperti Jumlah 2D dan Colok Bebas."""
    if df.empty or len(df) < 10:
        return {"jumlah_2d": "Data tidak cukup", "colok_bebas": "Data tidak cukup"}

    angka_str = df["angka"].astype(str).str.zfill(4)
    puluhan = angka_str.str[2].astype(int)
    satuan = angka_str.str[3].astype(int)
    jumlah = (puluhan + satuan) % 10
    jumlah_2d = ", ".join(map(str, jumlah.value_counts().nlargest(top_n).index))
    
    all_digits = "".join(angka_str.tolist())
    colok_bebas = ", ".join([item[0] for item in Counter(all_digits).most_common(top_n)])
    
    return {"jumlah_2d": jumlah_2d, "colok_bebas": colok_bebas}

def _get_ai_prediction_map(angka_list, start_digit_idx, top_n):
    """Helper function untuk membuat peta prediksi AI berdasarkan transisi digit."""
    transitions = defaultdict(list)
    for num_str in angka_list:
        start_digit = num_str[start_digit_idx]
        following_digits = [d for i, d in enumerate(num_str) if i != start_digit_idx]
        transitions[start_digit].extend(following_digits)

    prediction_map = {}
    for start_digit, following_digits in transitions.items():
        top_digits_counts = Counter(following_digits).most_common()
        
        final_digits = list(dict.fromkeys([d for d, c in top_digits_counts]))
        
        if len(final_digits) < top_n:
            all_possible_digits = list(map(str, range(10)))
            random.shuffle(all_possible_digits)
            current_digits_set = set(final_digits)
            
            for digit in all_possible_digits:
                if len(final_digits) >= top_n:
                    break
                if digit not in current_digits_set:
                    final_digits.append(digit)

        prediction_map[start_digit] = "".join(final_digits[:top_n])
        
    return prediction_map

def calculate_markov_ai(df, top_n=6, mode='belakang'):
    """Menghitung AI berdasarkan transisi dari posisi digit yang dipilih."""
    if df.empty or len(df) < 10:
        return "Data tidak cukup untuk analisis."

    mode_to_idx = {'depan': 3, 'tengah': 1, 'belakang': 0}
    if mode not in mode_to_idx:
        return f"Mode analisis tidak valid: {mode}"
    
    start_idx = mode_to_idx[mode]

    angka_str_list = df["angka"].astype(str).str.zfill(4).tolist()
    prediction_map = _get_ai_prediction_map(angka_str_list, start_digit_idx=start_idx, top_n=top_n)
    
    output_lines = []
    for num_str in angka_str_list[-30:]:
        start_digit = num_str[start_idx]
        ai = prediction_map.get(start_digit, "")
        output_lines.append(f"{num_str} = {ai} ai")
    
    return "\n".join(output_lines)

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
    
    labels_to_process = DIGIT_LABELS + BBFS_LABELS + JUMLAH_LABELS + SHIO_LABELS
    sequences, targets = [], {label: [] for label in labels_to_process}
    
    for i in range(len(angka) - window_size):
        window = [str(x).zfill(4) for x in angka[i:i+window_size+1]]
        if any(not x.isdigit() for x in window): continue
        sequences.append([int(d) for num in window[:-1] for d in num])
        target_digits = [int(d) for d in window[-1]]
        
        for j, label in enumerate(DIGIT_LABELS):
            targets[label].append(to_categorical(target_digits[j], num_classes=10))
        
        jumlah_map = {
            "jumlah_depan": (target_digits[0] + target_digits[1]) % 10,
            "jumlah_tengah": (target_digits[1] + target_digits[2]) % 10,
            "jumlah_belakang": (target_digits[2] + target_digits[3]) % 10,
        }
        for label, value in jumlah_map.items():
            targets[label].append(to_categorical(value, num_classes=10))

        bbfs_map = {
            "bbfs_ribuan-ratusan": [target_digits[0], target_digits[1]],
            "bbfs_ratusan-puluhan": [target_digits[1], target_digits[2]],
            "bbfs_puluhan-satuan": [target_digits[2], target_digits[3]],
        }
        for label, digit_pair in bbfs_map.items():
            multi_hot_target = np.zeros(10, dtype=np.float32)
            for digit in np.unique(digit_pair):
                multi_hot_target[digit] = 1.0
            targets[label].append(multi_hot_target)

        # Logika untuk Shio (1-12)
        shio_num_map = {
            "shio_depan": target_digits[0] * 10 + target_digits[1],
            "shio_tengah": target_digits[1] * 10 + target_digits[2],
            "shio_belakang": target_digits[2] * 10 + target_digits[3],
        }
        for label, two_digit_num in shio_num_map.items():
            shio_index = (two_digit_num - 1) % 12 if two_digit_num > 0 else 11
            targets[label].append(to_categorical(shio_index, num_classes=12))

    final_targets = {label: np.array(v) for label, v in targets.items() if v}
    return np.array(sequences), final_targets

def build_model(input_len, model_type="lstm", problem_type="multiclass", num_classes=10):
    """Membangun arsitektur model AI, mendukung jumlah kelas yang dinamis."""
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

    if problem_type == "multilabel":
        outputs = Dense(num_classes, activation='sigmoid')(x)
        loss = "binary_crossentropy"
    else: 
        outputs = Dense(num_classes, activation='softmax')(x)
        loss = "categorical_crossentropy"

    model = Model(inputs, outputs)
    return model, loss


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

def find_best_window_size(df, label, model_type, min_ws, max_ws, top_n, top_n_shio):
    """Mencari window size terbaik, mendukung semua jenis kategori."""
    best_ws, best_score = None, -1
    table_data = []
    
    if label in BBFS_LABELS:
        problem_type = "multilabel"
    elif label in SHIO_LABELS:
        problem_type = "shio"
    else:
        problem_type = "multiclass"

    k_val = top_n_shio if problem_type == "shio" else top_n
    num_classes = 12 if problem_type == "shio" else 10
    
    progress_bar = st.progress(0.0, text=f"Memulai scan untuk {label.upper()}...")
    total_steps = max(1, max_ws - min_ws + 1)
    
    for i, ws in enumerate(range(min_ws, max_ws + 1)):
        progress_bar.progress((i + 1) / total_steps, text=f"Mencoba WS={ws} untuk {label.upper()}...")
        try:
            X, y_dict = preprocess_data(df, window_size=ws)
            if label not in y_dict or y_dict[label].shape[0] < 10: continue
            y = y_dict[label]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            model, loss_function = build_model(X.shape[1], model_type, problem_type, num_classes)
            
            metrics = ['accuracy'] 
            if problem_type != 'multilabel':
                metrics.append(TopKCategoricalAccuracy(k=k_val))
            
            model.compile(optimizer="adam", loss=loss_function, metrics=metrics)
            
            model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=0)
            
            eval_results = model.evaluate(X_val, y_val, verbose=0)
            preds = model.predict(X_val, verbose=0)
            avg_conf = np.mean(np.sort(preds, axis=1)[:, -k_val:]) * 100
            last_pred = model.predict(X[-1:], verbose=0)[0]
            top_indices = np.argsort(last_pred)[::-1][:k_val]

            if problem_type == "shio":
                top_n_pred_str = ", ".join(map(str, top_indices + 1))
            else:
                top_n_pred_str = ", ".join(map(str, top_indices))


            if problem_type == 'multilabel':
                acc = eval_results[1]
                score = (acc * 0.7) + (avg_conf / 100 * 0.3)
                top_n_acc_display = "N/A"
            else: 
                acc = eval_results[1]
                top_n_acc = eval_results[2]
                score = (acc * 0.2) + (top_n_acc * 0.5) + (avg_conf / 100 * 0.3)
                top_n_acc_display = f"{top_n_acc*100:.2f}"

            table_data.append((ws, f"{acc*100:.2f}", top_n_acc_display, f"{avg_conf:.2f}", top_n_pred_str))
            if score > best_score:
                best_score, best_ws = score, ws
        except Exception as e:
            st.error(f"Gagal saat scan {label} di WS={ws}: {e}")
            continue
            
    progress_bar.empty()
    if not table_data: return None, None
    return best_ws, pd.DataFrame(table_data, columns=["Window Size", "Acc (%)", f"Top-{k_val} Acc (%)", "Conf (%)", f"Top-{k_val}"])


def train_and_save_model(df, lokasi, window_dict, model_type):
    """Melatih dan menyimpan model untuk setiap posisi digit (hanya untuk multi-class)."""
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

        model, loss = build_model(X.shape[1], model_type, problem_type='multiclass', num_classes=10)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

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
st.caption("editing by: Andi Prediction")

try: from lokasi_list import lokasi_list
except ImportError: lokasi_list = ["BULLSEYE", "HONGKONG", "SYDNEY", "SINGAPORE"]

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
    jumlah_digit = st.slider("🔢 Jumlah Digit Prediksi", 1, 9, 6)
    jumlah_digit_shio = st.slider("🐉 Jumlah Digit Prediksi Khusus Shio", 1, 12, 12)
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
tab_scan, tab_manajemen, tab_angka_main, tab_prediksi = st.tabs([
    "🪟 Scan Window Size",
    "⚙️ Manajemen Model",
    "🎯 Angka Main",
    "🔮 Prediksi & Hasil"
])

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
                total_kombinasi = np.prod([len(r) for r in result])
                st.subheader(f"🔢 Semua Kombinasi 4D ({total_kombinasi} Line)")

                if total_kombinasi > 5000:
                    st.warning(f"Jumlah kombinasi ({total_kombinasi}) sangat besar.")
                
                all_combinations = list(product(*result))
                kombinasi_4d_list = ["".join(map(str, combo)) for combo in all_combinations]
                output_string = " * ".join(kombinasi_4d_list)
                st.text_area(f"Total {total_kombinasi} Kombinasi Penuh", output_string, height=300)

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
    if st.button("📚 Latih & Simpan Semua Model AI", use_container_width=True, type="primary"):
        max_ws = max(window_per_digit.values())
        if len(df) < max_ws + 10:
            st.error(f"Data tidak cukup. Butuh {max_ws + 10} baris.")
        else:
            train_and_save_model(df, selected_lokasi, window_per_digit, model_type)
            st.success("✅ Semua model berhasil dilatih!"); st.rerun()

with tab_scan:
    st.subheader("Pencarian Window Size (WS) Optimal per Kategori")
    st.info("Klik tombol scan untuk setiap kategori.")
    scan_cols = st.columns(2)
    
    min_ws = scan_cols[0].number_input("Min WS", 1, 99, 1)
    max_ws = scan_cols[1].number_input("Max WS", 1, 100, 25)

    scan_ready = True
    if min_ws >= max_ws:
        st.warning(f"'Min WS' ({min_ws}) harus lebih kecil dari 'Max WS' ({max_ws}).")
        scan_ready = False

    if st.button("❌ Hapus Hasil Scan"):
        st.session_state.scan_outputs = {}
        st.rerun()
    st.divider()
    
    SCAN_LABELS = DIGIT_LABELS + JUMLAH_LABELS + BBFS_LABELS + SHIO_LABELS
    
    def display_scan_button(label, columns):
        display_label = label.replace('_', ' ').upper()
        if columns.button(f"🔎 Scan {display_label}", use_container_width=True, disabled=not scan_ready, key=f"scan_{label}"):
            if len(df) < max_ws + 10: st.error(f"Data tidak cukup. Butuh {max_ws + 10} baris.")
            else:
                st.toast(f"Memindai {display_label}...", icon="⏳"); st.session_state.scan_outputs[label] = "PENDING"; st.rerun()
    
    st.markdown("**Kategori Digit**")
    for i, label in enumerate(DIGIT_LABELS):
        display_scan_button(label, st.columns(len(DIGIT_LABELS))[i])

    st.markdown("**Kategori Jumlah**")
    for i, label in enumerate(JUMLAH_LABELS):
        display_scan_button(label, st.columns(len(JUMLAH_LABELS))[i])

    st.markdown("**Kategori BBFS**")
    for i, label in enumerate(BBFS_LABELS):
        display_scan_button(label, st.columns(len(BBFS_LABELS))[i])

    st.markdown("**Kategori Shio**")
    for i, label in enumerate(SHIO_LABELS):
        display_scan_button(label, st.columns(len(SHIO_LABELS))[i])
    
    st.divider()

    for label in [l for l in SCAN_LABELS if l in st.session_state.scan_outputs]:
        data = st.session_state.scan_outputs[label]
        expander_title = f"Hasil Scan untuk {label.replace('_', ' ').upper()}"
        with st.expander(expander_title, expanded=True):
            if data == "PENDING":
                best_ws, result_table = find_best_window_size(df, label, model_type, min_ws, max_ws, top_n=jumlah_digit, top_n_shio=jumlah_digit_shio)
                st.session_state.scan_outputs[label] = {"ws": best_ws, "table": result_table}
                st.rerun()
            
            elif isinstance(data, dict):
                result_df = data.get("table")
                if result_df is not None and not result_df.empty:
                    st.dataframe(result_df)
                    st.markdown("---")
                    
                    prediction_column_name = next((col for col in result_df.columns if col.startswith("Top-") and "Acc" not in col), None)

                    if prediction_column_name:
                        st.markdown(f"👇 **Salin Hasil dari Kolom {prediction_column_name}**")
                        copyable_text = "\n".join(result_df[prediction_column_name].astype(str))
                        st.text_area(
                            f"Klik untuk menyalin",
                            value=copyable_text,
                            height=250, key=f"copy_{label}"
                        )

                    if data["ws"] is not None:
                        st.success(f"✅ WS terbaik: {data['ws']}")
                    else:
                        st.warning("Tidak ditemukan WS yang menonjol.")
                else:
                    st.warning("Tidak ada hasil untuk rentang WS ini.")

with tab_angka_main:
    st.subheader("Analisis Angka Main dari Data Historis")
    
    if len(df) < 10:
        st.warning("Data historis tidak cukup (minimal 10 baris).")
    else:
        col1, col2 = st.columns([2, 1]) 
        with col1:
            st.markdown("##### Analisis AI Berdasarkan Posisi")

            with st.expander("Analisis AI Depan (berdasarkan digit EKOR/Satuan)", expanded=True):
                with st.spinner("Menganalisis AI Depan..."):
                    result_depan = calculate_markov_ai(df, top_n=jumlah_digit, mode='depan')
                st.text_area(
                    "Hasil Analisis (Depan)", result_depan, height=300,
                    label_visibility="collapsed", key="ai_depan"
                )

            with st.expander("Analisis AI Tengah (berdasarkan digit AS/Ratusan)"):
                with st.spinner("Menganalisis AI Tengah..."):
                    result_tengah = calculate_markov_ai(df, top_n=jumlah_digit, mode='tengah')
                st.text_area(
                    "Hasil Analisis (Tengah)", result_tengah, height=300,
                    label_visibility="collapsed", key="ai_tengah"
                )
            
            with st.expander("Analisis AI Belakang (berdasarkan digit KOP/Ribuan)"):
                with st.spinner("Menganalisis AI Belakang..."):
                    result_belakang = calculate_markov_ai(df, top_n=jumlah_digit, mode='belakang')
                st.text_area(
                    "Hasil Analisis (Belakang)", result_belakang, height=300,
                    label_visibility="collapsed", key="ai_belakang"
                )

        with col2:
            st.markdown("##### Statistik Lainnya")
            with st.spinner("Menghitung statistik..."):
                stats = calculate_angka_main_stats(df, top_n=jumlah_digit)
            
            st.markdown(f"**Jumlah 2D (Belakang):**")
            st.code(stats['jumlah_2d'])
            
            st.markdown(f"**Colok Bebas:**")
            st.code(stats['colok_bebas'])

    st.info("Angka di atas adalah hasil analisis statistik dari data historis dan bukan jaminan hasil.")
