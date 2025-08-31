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

# --- FUNGSI HELPER UNTUK TENSORFLOW (LAZY LOADING) ---
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

# --- FUNGSI-FUNGSI LAINNYA ---
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

def find_best_window_size(df, label, model_type, min_ws, max_ws, top_n, top_n_shio):
    # Impor library berat hanya di dalam fungsi ini
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.metrics import TopKCategoricalAccuracy
    from tensorflow.keras.utils import to_categorical

    # Fungsi-fungsi helper yang membutuhkan TF juga didefinisikan di sini
    def build_tf_model(input_len, model_type, problem_type, num_classes):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
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
            # (Fungsi tf_preprocess_data dan variasinya dimasukkan ke sini)
            # Ini memastikan semua kode terkait TF terisolasi
            if is_jalur_scan: 
                # Logika preprocess untuk jalur...
                pass # Placeholder
            else: 
                # Logika preprocess umum...
                pass # Placeholder
            
            # Placeholder untuk logika training & evaluasi
            score = random.random() # Ganti dengan skor asli Anda
            pred_str = ", ".join(map(str, random.sample(range(10), 4))) # Ganti dengan prediksi asli
            all_scores.append((score, ws))
            if is_jalur_scan: table_data.append((ws, pred_str, "Jalur X => ..."))
            else: table_data.append((ws, pred_str))
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

# ... (UI sidebar dan komponen lainnya tetap sama) ...
# Placeholder untuk UI Anda
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", ["BULLSEYE", "HONGKONG LOTTO"])
    putaran = st.number_input("🔁 Jumlah Putaran Terakhir", 10, 1000, 100)
    jumlah_digit = st.slider("🔢 Jumlah Digit Prediksi", 1, 9, 9)
    jumlah_digit_shio = st.slider("🐉 Jumlah Digit Prediksi Khusus Shio", 1, 12, 12)

df = pd.DataFrame({"angka": st.session_state.get("angka_list", [])})
# PERBAIKAN: Logika baru untuk mem-parsing input manual
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
                    if num_str.isdigit():
                        new_angka_list.append(num_str)
                except IndexError:
                    continue
            elif len(cleaned_line) >= 4 and cleaned_line[:4].isdigit():
                new_angka_list.append(cleaned_line[:4])
        st.session_state.angka_list = new_angka_list
        st.rerun()

# --- TAB SCAN WINDOW SIZE ---
with st.tabs(["Scan Window Size"])[0]:
    st.subheader("Pencarian Window Size (WS) Optimal per Kategori")
    scan_cols = st.columns(2)
    min_ws = scan_cols[0].number_input("Min WS", 1, 99, 5)
    max_ws = scan_cols[1].number_input("Max WS", 1, 100, 31)
    if st.button("❌ Hapus Hasil Scan"): st.session_state.scan_outputs = {}; st.rerun()
    st.divider()

    def create_scan_button(label, container):
        if container.button(f"🔎 Scan {label.replace('_', ' ').upper()}", key=f"scan_{label}", use_container_width=True):
            if len(df) < max_ws + 10:
                st.error("Data tidak cukup.")
            else:
                st.toast(f"Memulai scan untuk {label.replace('_', ' ').upper()}...", icon="⏳")
                # Panggil fungsi berat di sini, di dalam button click
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
                # PERBAIKAN: Memastikan hasil WS selalu dalam bentuk list untuk mencegah TypeError
                top_ws_list = data.get("ws")
                if top_ws_list:
                    if not isinstance(top_ws_list, list): # Jika hanya satu angka, ubah jadi list
                        top_ws_list = [top_ws_list]
                    display_string = "\n".join([f"✅ WS terbaik: {ws}" for ws in top_ws_list])
                    st.success(display_string)
            else:
                st.warning("Tidak ada hasil untuk rentang WS ini.")
