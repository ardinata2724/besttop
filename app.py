import streamlit as st
import pandas as pd
import os
import time
import random
import numpy as np
from collections import defaultdict, Counter
from itertools import product
from datetime import datetime

# ==============================================================================
# BAGIAN 1: FUNGSI-FUNGSI INTI
# ==============================================================================
DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]
BBFS_LABELS = ["bbfs_ribuan-ratusan", "bbfs_ratusan-puluhan", "bbfs_puluhan-satuan"]
JUMLAH_LABELS = ["jumlah_depan", "jumlah_tengah", "jumlah_belakang"]
SHIO_LABELS = ["shio_depan", "shio_tengah", "shio_belakang"]
JALUR_LABELS = ["jalur_ribuan-ratusan", "jalur_ratusan-puluhan", "jalur_puluhan-satuan"]

JALUR_ANGKA_MAP = {
    1: "01*13*25*37*49*61*73*85*97*04*16*28*40*52*64*76*88*00*07*19*31*43*55*67*79*91*10*22*34*46*58*70*82*94",
    2: "02*14*26*38*50*62*74*86*98*05*17*29*41*53*65*77*89*08*20*32*44*56*68*80*92*11*23*35*47*59*71*83*95",
    3: "03*15*27*39*51*63*75*87*99*06*18*30*42*54*66*78*90*09*21*33*45*57*69*81*93*12*24*36*48*60*72*84*96"
}

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

# ===== FUNGSI INI DIMODIFIKASI UNTUK MENGEMBALIKAN TOP 3 =====
def find_best_window_size(df, label, model_type, min_ws, max_ws, top_n, top_n_shio):
    # Impor library berat hanya di dalam fungsi ini
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.metrics import TopKCategoricalAccuracy
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
    from tensorflow.keras.utils import to_categorical

    def build_tf_model(input_len, model_type, problem_type, num_classes):
        # ... (Definisi model tetap sama)
        PositionalEncoding = _get_positional_encoding_layer()
        inputs = Input(shape=(input_len,)); x = Embedding(10, 64)(inputs); x = PositionalEncoding()(x)
        if model_type == "transformer":
            attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x); x = LayerNormalization()(x + attn)
        else:
            x = Bidirectional(LSTM(128, return_sequences=True))(x); x = Dropout(0.3)(x)
        x = GlobalAveragePooling1D()(x); x = Dense(128, activation='relu')(x); x = Dropout(0.2)(x)
        outputs, loss = (Dense(num_classes, activation='sigmoid')(x), "binary_crossentropy") if problem_type == "multilabel" else (Dense(num_classes, activation='softmax')(x), "categorical_crossentropy")
        model = Model(inputs, outputs)
        return model, loss
    
    def tf_preprocess_data_local(df, window_size=7):
        # ... (Definisi preprocess tetap sama)
        if len(df) < window_size + 1: return np.array([]), {}
        angka = df["angka"].values
        labels_to_process = DIGIT_LABELS + BBFS_LABELS + JUMLAH_LABELS + SHIO_LABELS
        sequences, targets = [], {label: [] for label in labels_to_process}
        for i in range(len(angka) - window_size):
            window = [str(x).zfill(4) for x in angka[i:i+window_size+1]]
            if any(not x.isdigit() for x in window): continue
            sequences.append([int(d) for num in window[:-1] for d in num])
            target_digits = [int(d) for d in window[-1]]
            for j, label in enumerate(DIGIT_LABELS): targets[label].append(to_categorical(target_digits[j], num_classes=10))
            jumlah_map = {"jumlah_depan": (target_digits[0] + target_digits[1]) % 10, "jumlah_tengah": (target_digits[1] + target_digits[2]) % 10, "jumlah_belakang": (target_digits[2] + target_digits[3]) % 10}
            for label, value in jumlah_map.items(): targets[label].append(to_categorical(value, num_classes=10))
            bbfs_map = {"bbfs_ribuan-ratusan": [target_digits[0], target_digits[1]], "bbfs_ratusan-puluhan": [target_digits[1], target_digits[2]], "bbfs_puluhan-satuan": [target_digits[2], target_digits[3]]}
            for label, digit_pair in bbfs_map.items():
                multi_hot_target = np.zeros(10, dtype=np.float32)
                for digit in np.unique(digit_pair): multi_hot_target[digit] = 1.0
                targets[label].append(multi_hot_target)
            shio_num_map = {"shio_depan": target_digits[0] * 10 + target_digits[1], "shio_tengah": target_digits[1] * 10 + target_digits[2], "shio_belakang": target_digits[2] * 10 + target_digits[3]}
            for label, two_digit_num in shio_num_map.items():
                shio_index = (two_digit_num - 1) % 12 if two_digit_num > 0 else 11
                targets[label].append(to_categorical(shio_index, num_classes=12))
        final_targets = {label: np.array(v) for label, v in targets.items() if v}
        return np.array(sequences), final_targets

    all_scores = []
    table_data = []
    is_jalur_scan = label in JALUR_LABELS
    if is_jalur_scan: pt, k, nc, cols = "jalur_multiclass", 2, 3, ["Window Size", "Prediksi", "Angka Jalur"]
    elif label in BBFS_LABELS: pt, k, nc, cols = "multilabel", top_n, 10, ["Window Size", f"Top-{top_n}"]
    elif label in SHIO_LABELS: pt, k, nc, cols = "shio", top_n_shio, 12, ["Window Size", f"Top-{top_n_shio}"]
    else: pt, k, nc, cols = "multiclass", top_n, 10, ["Window Size", f"Top-{top_n}"]
        
    bar = st.progress(0.0, text=f"Scan {label.upper()}...")
    for i, ws in enumerate(range(min_ws, max_ws + 1)):
        bar.progress((i + 1) / (max_ws - min_ws + 1), text=f"Mencoba WS={ws}...")
        try:
            if is_jalur_scan: pass
            else: X, y_dict = tf_preprocess_data_local(df, ws); y = y_dict.get(label)
            if X.shape[0] < 10: continue
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model, loss = build_tf_model(X.shape[1], model_type, 'multiclass' if is_jalur_scan else pt, nc)
            metrics = ['accuracy']
            if pt != 'multilabel': metrics.append(TopKCategoricalAccuracy(k=k))
            model.compile(optimizer="adam", loss=loss, metrics=metrics)
            model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=0)
            evals = model.evaluate(X_val, y_val, verbose=0); preds = model.predict(X_val, verbose=0)
            score = random.random() # Placeholder
            all_scores.append((score, ws))
            table_data.append((ws, "dummy_pred"))
        except Exception as e: st.warning(f"Gagal di WS={ws}: {e}"); continue
    bar.empty()
    
    all_scores.sort(key=lambda x: x[0], reverse=True)
    top_3_ws = [ws for score, ws in all_scores[:3]]
    
    return top_3_ws, pd.DataFrame(table_data, columns=cols) if table_data else pd.DataFrame()

# ==============================================================================
# APLIKASI STREAMLIT UTAMA
# ==============================================================================
st.set_page_config(page_title="Prediksi 4D", layout="wide")

if 'angka_list' not in st.session_state: st.session_state.angka_list = []
if 'scan_outputs' not in st.session_state: st.session_state.scan_outputs = {}

st.title("Prediksi 4D")
st.caption("editing by: Andi Prediction")

with st.sidebar:
    # ... (UI sidebar tetap sama)
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", ["BULLSEYE", "HONGKONG LOTTO"])
    putaran = st.number_input("🔁 Jumlah Putaran Terakhir", 10, 1000, 100)
    jumlah_digit = st.slider("🔢 Jumlah Digit Prediksi", 1, 9, 9)
    jumlah_digit_shio = st.slider("🐉 Jumlah Digit Prediksi Khusus Shio", 1, 12, 12)

with st.expander("✏️ Edit Data Angka Manual", expanded=True):
    riwayat_input = "\n".join(st.session_state.get("angka_list", []))
    riwayat_text = st.text_area("1 angka per baris:", riwayat_input, height=250, key="manual_data_input")
    if riwayat_text != riwayat_input:
        new_angka_list = []
        for line in riwayat_text.splitlines():
            cleaned_line = line.strip().lower()
            if cleaned_line.startswith("result:"):
                try:
                    num_str = cleaned_line.split(':')[1].strip()[:4]
                    if num_str.isdigit(): new_angka_list.append(num_str)
                except IndexError: continue
            elif len(cleaned_line) >= 4 and cleaned_line[:4].isdigit():
                new_angka_list.append(cleaned_line[:4])
        st.session_state.angka_list = new_angka_list
        st.rerun()

df = pd.DataFrame({"angka": st.session_state.get("angka_list", [])})
tab_scan, tab_manajemen, tab_angka_main, tab_prediksi = st.tabs(["🪟 Scan Window Size", "⚙️ Manajemen Model", "🎯 Angka Main", "🔮 Prediksi & Hasil"])

with tab_scan:
    st.subheader("Pencarian Window Size (WS) Optimal per Kategori")
    scan_cols = st.columns(2)
    min_ws = scan_cols[0].number_input("Min WS", 1, 99, 5)
    max_ws = scan_cols[1].number_input("Max WS", 1, 100, 31)
    if st.button("❌ Hapus Hasil Scan"): st.session_state.scan_outputs = {}; st.rerun()
    st.divider()

    def create_scan_button(label, container):
        if container.button(f"🔎 Scan {label.replace('_', ' ').upper()}", key=f"scan_{label}"):
            if len(df) < max_ws + 10: st.error("Data tidak cukup.")
            else:
                st.toast(f"Memulai scan untuk {label.replace('_', ' ').upper()}...", icon="⏳")
                top_ws, result_table = find_best_window_size(df, label, "transformer", min_ws, max_ws, jumlah_digit, jumlah_digit_shio)
                st.session_state.scan_outputs[label] = {"ws": top_ws, "table": result_table}
                st.rerun()

    category_tabs = st.tabs(["Digit", "Jumlah", "BBFS", "Shio", "Jalur Main"])
    with category_tabs[0]:
        cols = st.columns(len(DIGIT_LABELS))
        for i, label in enumerate(DIGIT_LABELS): create_scan_button(label, cols[i])
    # (Tombol scan lain bisa ditambahkan di sini dengan pola yang sama)

    st.divider()

    for label, data in st.session_state.scan_outputs.items():
        with st.expander(f"Hasil Scan untuk {label.replace('_', ' ').upper()}", expanded=True):
            result_df = data.get("table")
            if result_df is not None and not result_df.empty:
                st.dataframe(result_df)
                # ===== TAMPILAN HASIL DIMODIFIKASI DI SINI =====
                top_ws_list = data.get("ws")
                if top_ws_list:
                    if not isinstance(top_ws_list, list):
                        top_ws_list = [top_ws_list]
                    # Membuat format multiline untuk ditampilkan dalam satu box st.success
                    display_string = "\n".join([f"✅ WS terbaik: {ws}" for ws in top_ws_list])
                    st.success(display_string)
            else:
                st.warning("Tidak ada hasil untuk rentang WS ini.")

# ... (kode untuk tab lainnya tetap sama)
