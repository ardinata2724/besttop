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
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import time
import random
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from sklearn.model_selection import KFold
from itertools import product
from markov_model import top6_markov

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

class PositionalEncoding(tf.keras.layers.Layer):
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

# --- FUNGSI YANG HILANG DITAMBAHKAN KEMBALI ---
def build_transformer_model(input_len, embed_dim=32, heads=4, temperature=0.5):
    inputs = Input(shape=(input_len,))
    x = Embedding(input_dim=10, output_dim=embed_dim)(inputs)
    x = PositionalEncoding()(x)
    for _ in range(2):
        attn = MultiHeadAttention(num_heads=heads, key_dim=embed_dim)(x, x)
        x = LayerNormalization()(x + attn)
        ff = Dense(embed_dim, activation='relu')(x)
        x = LayerNormalization()(x + ff)
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    logits = Dense(10)(x)
    outputs = tf.keras.layers.Activation('softmax')(logits / temperature)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_save_model(df, lokasi, window_dict, model_type="lstm"):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)
    for label in DIGIT_LABELS:
        window_size = window_dict.get(label, 7)
        if len(df) < window_size + 5:
            continue
        
        X, y_dict = preprocess_data(df, window_size=window_size)
        if label not in y_dict or y_dict[label].shape[0] < 10: # Pastikan data cukup untuk split
            continue
        
        y = y_dict[label]

        loc_id = lokasi.lower().strip().replace(" ", "_")
        log_path = f"training_logs/history_{loc_id}_{label}_{model_type}.csv"
        model_path = f"saved_models/{loc_id}_{label}_{model_type}.h5"
        
        if os.path.exists(model_path):
            model = load_model(model_path, compile=True, custom_objects={"PositionalEncoding": PositionalEncoding})
        else:
            model = build_transformer_model(X.shape[1]) if model_type == "transformer" else build_lstm_model(X.shape[1])

        # --- PERBAIKAN LOGIKA TRAINING UNTUK MENGHINDARI WARNING ---
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        callbacks = [
            CSVLogger(log_path),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        ]

        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, 
                  validation_data=(X_val, y_val), 
                  callbacks=callbacks)
        model.save(model_path)


def top6_model(df, lokasi=None, model_type="lstm", return_probs=False, temperature=0.5, window_dict=None, mode_prediksi="hybrid", threshold=0.001, top_n=6):
    results, probs = [], []
    loc_id = lokasi.lower().replace(" ", "_")
    for label in DIGIT_LABELS:
        window_size = window_dict.get(label, 7)
        X, _ = preprocess_data(df, window_size=window_size)
        if X.shape[0] == 0:
            return None # Kembalikan None jika tidak ada data
        
        path = f"saved_models/{loc_id}_{label}_{model_type}.h5"
        if not os.path.exists(path):
            return None # Kembalikan None jika model tidak ada
        
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
                top_n_digits = [d for d, score in sorted_scores if avg[d] >= threshold][:top_n]

            # Pastikan jumlahnya N
            final_digits = list(dict.fromkeys(top_n_digits))[:top_n]
            current_len = len(final_digits)
            if current_len < top_n:
                 all_possible = list(range(10))
                 random.shuffle(all_possible)
                 for digit in all_possible:
                     if len(final_digits) >= top_n:
                         break
                     if digit not in final_digits:
                         final_digits.append(digit)

            results.append(final_digits)
            probs.append([avg[d] for d in final_digits])

        except Exception as e:
            print(f"[ERROR {label}] {e}")
            return None
            
    return (results, probs) if return_probs else results

# ... Sisa fungsi di ai_model.py (seperti kombinasi_4d, top6_ensemble, dll) tetap sama seperti sebelumnya ...
# (Untuk keringkasan, saya tidak menyertakan ulang sisa kode yang tidak berubah)
# Pastikan Anda menggunakan versi lengkap dari file ai_model.py yang saya berikan di respons sebelumnya.
