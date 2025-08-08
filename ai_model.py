import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dropout, Dense,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import os
import pandas as pd
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from sklearn.model_selection import KFold
from itertools import product
from markov_model import top6_markov

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

class PositionalEncoding(tf.keras.layers.Layer):
    # ... (Tidak ada perubahan di kelas ini)
    def call(self, x):
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], dtype=tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], dtype=tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        return x + tf.cast(pos_encoding, tf.float32)


def preprocess_data(df, window_size=7):
    # ... (Tidak ada perubahan di fungsi ini)
    if len(df) < window_size + 1:
        return np.array([]), {label: np.array([]) for label in DIGIT_LABELS}
    
    angka = df["angka"].values
    total_data = len(angka)
    num_windows = (total_data - 1) // window_size
    start_index = total_data - (num_windows * window_size + 1)
    if start_index < 0:
        start_index = 0

    sequences = []
    targets = {label: [] for label in DIGIT_LABELS}

    for i in range(start_index, total_data - window_size):
        window = angka[i:i+window_size+1]
        if any(len(str(x)) != 4 or not str(x).isdigit() for x in window):
            continue
        seq = [int(d) for num in window[:-1] for d in f"{int(num):04d}"]
        sequences.append(seq)
        target_digits = [int(d) for d in f"{int(window[-1]):04d}"]
        for j, label in enumerate(DIGIT_LABELS):
            targets[label].append(to_categorical(target_digits[j], num_classes=10))
    
    X = np.array(sequences)
    y_dict = {label: np.array(targets[label]) for label in DIGIT_LABELS}
    return X, y_dict


def build_lstm_model(input_len, embed_dim=32, lstm_units=128, attention_heads=4, temperature=0.5):
    # ... (Tidak ada perubahan di fungsi ini)
    inputs = Input(shape=(input_len,), name="input_layer")
    x = Embedding(input_dim=10, output_dim=embed_dim, name="embedding")(inputs)
    x = PositionalEncoding()(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True), name="bilstm_1")(x)
    x = LayerNormalization(name="layernorm_1")(x)
    x = Dropout(0.3, name="dropout_1")(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True), name="bilstm_2")(x)
    x = LayerNormalization(name="layernorm_2")(x)
    x = MultiHeadAttention(num_heads=attention_heads, key_dim=embed_dim, name="multihead_attn")(x, x)
    x = Dropout(0.2, name="dropout_2")(x)
    x = GlobalAveragePooling1D(name="gap")(x)
    x = Dense(512, activation='relu', name="dense_1")(x)
    x = Dropout(0.3, name="dropout_3")(x)
    x = Dense(128, activation='relu', name="dense_2")(x)
    logits = Dense(10, name="logits")(x)
    outputs = tf.keras.layers.Activation('softmax', name="softmax")(logits / temperature)
    model = Model(inputs, outputs, name="lstm_digit_model")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_save_model(df, lokasi, window_dict, model_type="lstm"):
    # ... (Tidak ada perubahan di fungsi ini)
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    for label in DIGIT_LABELS:
        window_size = window_dict.get(label, 7)
        if len(df) < window_size + 5:
            continue
        
        X, y_dict = preprocess_data(df, window_size=window_size)
        y = y_dict[label]

        if X.shape[0] == 0 or y.shape[0] == 0:
            continue

        suffix = model_type
        loc_id = lokasi.lower().strip().replace(" ", "_")
        log_path = f"training_logs/history_{loc_id}_{label}_{suffix}.csv"
        model_path = f"saved_models/{loc_id}_{label}_{suffix}.h5"
        callbacks = [
            CSVLogger(log_path),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]

        if os.path.exists(model_path):
            model = load_model(model_path, compile=True, custom_objects={"PositionalEncoding": PositionalEncoding})
        else:
            # Model Transformer tidak didefinisikan di sini, jadi kita asumsikan LSTM saja
            model = build_lstm_model(X.shape[1])

        model.fit(X, y, epochs=50, batch_size=32, verbose=0, validation_split=0.2, callbacks=callbacks)
        model.save(model_path)


def top6_model(df, lokasi=None, model_type="lstm", return_probs=False, temperature=0.5, window_dict=None, mode_prediksi="hybrid", threshold=0.001, top_n=6):
    results, probs = [], []
    loc_id = lokasi.lower().replace(" ", "_")
    for label in DIGIT_LABELS:
        window_size = window_dict.get(label, 7)
        X, _ = preprocess_data(df, window_size=window_size)
        if X.shape[0] == 0:
            # Kembalikan list kosong yang sesuai dengan jumlah digit
            return [[], [], [], []], [[], [], [], []] if return_probs else [[], [], [], []]
        
        path = f"saved_models/{loc_id}_{label}_{model_type}.h5"
        if not os.path.exists(path):
            return None
        
        try:
            model = load_model(path, compile=False, custom_objects={"PositionalEncoding": PositionalEncoding})
            if model.input_shape[1] != X.shape[1]:
                return None
            
            pred = model.predict(X, verbose=0)
            avg = np.mean(pred, axis=0)
            avg /= np.sum(avg)
            
            top_n_digits = []
            if mode_prediksi == "confidence":
                top_n_digits = avg.argsort()[-top_n:][::-1]
            elif mode_prediksi == "ranked":
                score_dict = {i: (1.0 / (1 + rank)) for rank, i in enumerate(avg.argsort()[::-1])}
                top_n_digits_sorted = sorted(score_dict.items(), key=lambda x: -x[1])[:top_n]
                top_n_digits = [d for d, _ in top_n_digits_sorted]
            else: # Hybrid
                score_dict = {i: avg[i] * (1.0 / (1 + rank)) for rank, i in enumerate(avg.argsort()[::-1])}
                sorted_scores = sorted(score_dict.items(), key=lambda x: -x[1])
                # Ambil N digit teratas yang memenuhi threshold
                top_n_digits = [d for d, score in sorted_scores if avg[d] >= threshold][:top_n]

            # Pastikan jumlahnya N, jika kurang, tambahkan secara acak (opsional, tapi baik untuk konsistensi)
            current_digits = set(top_n_digits)
            all_possible = list(range(10))
            random.shuffle(all_possible)
            for digit in all_possible:
                if len(current_digits) >= top_n:
                    break
                if digit not in current_digits:
                    current_digits.add(digit)

            final_digits = list(current_digits)
            results.append(final_digits)
            probs.append([avg[d] for d in final_digits])

        except Exception as e:
            print(f"[ERROR {label}] {e}")
            return None
            
    return (results, probs) if return_probs else results

def kombinasi_4d(df, lokasi, model_type="lstm", top_n=10, min_conf=0.0001, power=1.5, mode='product', window_dict=None, mode_prediksi="hybrid", pred_n=6):
    result, probs = top6_model(df, lokasi=lokasi, model_type=model_type, return_probs=True,
                               window_dict=window_dict, mode_prediksi=mode_prediksi, top_n=pred_n)
    if result is None or probs is None or not all(result):
        return []
    
    combinations = list(product(*result))
    scores = []
    for combo in combinations:
        digit_scores = []
        valid = True
        for i in range(4):
            try:
                idx = result[i].index(combo[i])
                digit_scores.append(probs[i][idx] ** power)
            except ValueError:
                valid = False
                break
        if not valid:
            continue
        
        score = np.prod(digit_scores) if mode == 'product' else np.mean(digit_scores)
        if score >= min_conf:
            scores.append(("".join(map(str, combo)), score))
            
    return sorted(scores, key=lambda x: -x[1])[:top_n]

def top6_ensemble(df, lokasi, model_type="lstm", lstm_weight=0.6, markov_weight=0.4, window_dict=None, temperature=0.5, mode_prediksi="hybrid", top_n=6):
    lstm_result = top6_model(
        df,
        lokasi=lokasi,
        model_type=model_type,
        return_probs=False,
        window_dict=window_dict,
        temperature=temperature,
        mode_prediksi=mode_prediksi,
        top_n=top_n
    )
    
    markov_result, _ = top6_markov(df, top_n=top_n)
    
    if lstm_result is None or markov_result is None or not all(lstm_result) or not all(markov_result):
        return None

    ensemble = []
    for i in range(4):
        scores = {}
        
        # Beri skor berdasarkan posisi di list LSTM
        for rank, digit in enumerate(lstm_result[i]):
            scores[digit] = scores.get(digit, 0) + lstm_weight * (1.0 / (1 + rank))
            
        # Tambahkan skor dari list Markov
        for rank, digit in enumerate(markov_result[i]):
            scores[digit] = scores.get(digit, 0) + markov_weight * (1.0 / (1 + rank))

        top_n_digits = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        ensemble.append([x[0] for x in top_n_digits])
        
    return ensemble
    
def evaluate_lstm_accuracy_all_digits(df, lokasi, model_type="lstm", window_size=7, top_n=6):
    # ... (Fungsi ini sekarang menerima top_n)
    if isinstance(window_size, int):
        window_dict = {label: window_size for label in DIGIT_LABELS}
    else:
        window_dict = window_size

    acc_top1_list, acc_top_n_list = [], []
    loc_id = lokasi.lower().strip().replace(" ", "_")

    for label in DIGIT_LABELS:
        # ... (logika lainnya tetap sama)
        try:
            # ...
            acc_top1 = model.evaluate(X, y_true, verbose=0)[1]
            acc_top_n = evaluate_top_n_accuracy(model, X, y_true, n=top_n) # Panggil fungsi baru
            acc_top1_list.append(acc_top1)
            acc_top_n_list.append(acc_top_n)
        except Exception as e:
            # ...
            pass
    return acc_top1_list, acc_top_n_list, [] # return 3 elemen agar kompatibel

def evaluate_top_n_accuracy(model, X, y_true, n=6):
    """Menghitung akurasi top-n: apakah label benar termasuk dalam N prediksi teratas."""
    try:
        y_pred = model.predict(X, verbose=0)
        y_true_labels = np.argmax(y_true, axis=1)
        top_n_preds = np.argsort(y_pred, axis=1)[:, -n:]
        correct = np.array([
            true_label in top_n for true_label, top_n in zip(y_true_labels, top_n_preds)
        ])
        return np.mean(correct)
    except Exception as e:
        print(f"[ERROR evaluate_top_n_accuracy] {e}")
        return 0.0

def find_best_window_size_with_model_true(df, label, lokasi, model_type="lstm", min_ws=4, max_ws=20,
                                          temperature=1.0, use_cv=False, cv_folds=5, seed=42,
                                          min_acc=0.60, min_conf=0.60, top_n=6):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    best_ws = None
    best_score = 0

    table_data = []
    all_scores = []
    
    st.markdown(f"### 🔍 Pencarian Window Size - {label.upper()} (Top-{top_n})")
    status = st.empty()

    ws_range = list(range(min_ws, max_ws + 1))

    for idx, ws in enumerate(ws_range):
        try:
            status.info(f"🧠 Mencoba WS={ws} ({idx+1}/{len(ws_range)}) untuk **{label.upper()}**...")
            X, y_dict = preprocess_data(df, window_size=ws)
            if label not in y_dict or y_dict[label].shape[0] == 0 or X.shape[0] == 0:
                continue

            y = y_dict[label]
            acc_scores, top_n_acc_scores, conf_scores = [], [], []
            top_n_all = []

            # (Logika CV atau non-CV tetap sama, hanya metriknya yang diubah)
            model = build_lstm_model(X.shape[1]) # Sederhanakan untuk contoh
            model.compile(optimizer="adam",
                          loss="categorical_crossentropy",
                          metrics=["accuracy", TopKCategoricalAccuracy(k=top_n)]) # k dinamis
            model.fit(X, y, epochs=10, batch_size=32, verbose=0, validation_split=0.2,
                      callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
            
            eval_result = model.evaluate(X, y, verbose=0)
            val_acc = eval_result[1]
            top_n_acc = eval_result[2]

            preds = model.predict(X[-1:], verbose=0)[0]
            avg_conf = np.mean(np.sort(preds)[::-1][:top_n]) # ambil N teratas
            top_n_digits_pred = np.argsort(preds)[::-1][:top_n] # ambil N teratas
            
            if val_acc < min_acc or avg_conf < min_conf:
                continue

            score = (val_acc * 0.35) + (top_n_acc * 0.35) + (avg_conf * 0.30)

            table_data.append((ws, round(val_acc*100, 2), round(top_n_acc*100, 2), round(avg_conf*100, 2), top_n_digits_pred))
            all_scores.append((ws, val_acc, top_n_acc, avg_conf, top_n_digits_pred, score))

            if score > best_score:
                best_score = score
                best_ws = ws

        except Exception as e:
            st.warning(f"[GAGAL WS={ws}] {e}")
            continue

    status.info("✅ Selesai semua WS")

    # Ambil N digit teratas dari 5 WS terbaik
    top5_ws = sorted(all_scores, key=lambda x: -x[5])[:5]
    top_n_from_top5_ws = []
    for _, _, _, _, top_n_digits, _ in top5_ws:
        top_n_from_top5_ws.extend(top_n_digits)

    # Hitung frekuensi dan ambil N paling umum
    final_top_n_digits = [x[0] for x in Counter(top_n_from_top5_ws).most_common(top_n)]

    if table_data:
        df_table = pd.DataFrame(table_data, columns=["Window Size", "Acc (%)", f"Top-{top_n} Acc (%)", "Conf (%)", f"Top-{top_n}"])
        st.dataframe(df_table.sort_values("Window Size"))

    if not best_ws:
        st.error("Tidak ada Window Size yang memenuhi kriteria.")
        return None, []

    st.success(f"✅ {label.upper()} - WS terbaik: {best_ws}")
    
    return best_ws, final_top_n_digits
