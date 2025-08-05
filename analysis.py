import streamlit as st
import numpy as np
import pandas as pd
from ai_model import kombinasi_4d, find_best_window_size_with_model_true

# Definisikan ruang pencarian untuk setiap parameter
# Untuk mempercepat, kita gunakan rentang yang tidak terlalu besar
# Anda bisa menyesuaikannya jika perlu
TEMPERATURE_RANGE = np.arange(0.4, 1.6, 0.4)  # Rentang: 0.4, 0.8, 1.2
POWER_RANGE = np.arange(1.0, 3.1, 1.0)      # Rentang: 1.0, 2.0, 3.0
MIN_CONF_RANGE = [0.0005, 0.001, 0.005]     # Rentang: 0.0005, 0.001, 0.005

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

def run_analysis(df, selected_lokasi, model_type, progress_callback):
    """
    Menjalankan analisis untuk menemukan pengaturan terbaik.

    Args:
        df (pd.DataFrame): DataFrame dengan data angka.
        selected_lokasi (str): Lokasi pasaran yang dipilih.
        model_type (str): Tipe model ('lstm' atau 'transformer').
        progress_callback (function): Fungsi untuk update progress bar dan teks.

    Returns:
        dict: Pengaturan terbaik yang ditemukan.
    """
    best_score = -1
    best_params = {}
    
    st.info("Mulai Analisa... Ini mungkin memakan waktu beberapa menit.")

    # --- Langkah 1: Cari Window Size Terbaik (dijalankan sekali) ---
    progress_callback(0.1, "Mencari Window Size terbaik untuk setiap digit...")
    best_window_sizes = {}
    for i, label in enumerate(DIGIT_LABELS):
        progress_callback(0.1 + (i * 0.1), f"Menganalisa Window Size untuk {label.upper()}...")
        try:
            # Menggunakan fungsi yang sudah ada untuk mencari WS
            # Kita gunakan parameter default agar tidak terlalu lama
            ws, _ = find_best_window_size_with_model_true(
                df, label, selected_lokasi, model_type,
                min_ws=5, max_ws=15, use_cv=False, min_acc=0.5, min_conf=0.5
            )
            best_window_sizes[label] = ws if ws else 7 # default 7 jika tidak ditemukan
        except Exception as e:
            st.warning(f"Gagal mencari WS untuk {label}: {e}. Menggunakan default (7).")
            best_window_sizes[label] = 7
    
    st.success(f"Window Size terbaik ditemukan: {best_window_sizes}")
    best_params["window_per_digit"] = best_window_sizes

    # --- Langkah 2: Cari Parameter Lainnya ---
    total_combinations = len(TEMPERATURE_RANGE) * len(POWER_RANGE) * len(MIN_CONF_RANGE)
    current_combination = 0

    for temp in TEMPERATURE_RANGE:
        for power in POWER_RANGE:
            for min_conf in MIN_CONF_RANGE:
                current_combination += 1
                progress = 0.5 + (current_combination / total_combinations * 0.5)
                progress_callback(progress, f"Menguji kombinasi {current_combination}/{total_combinations}...")

                try:
                    # Dapatkan kombinasi 4D dengan parameter saat ini
                    top_komb = kombinasi_4d(
                        df,
                        lokasi=selected_lokasi,
                        model_type=model_type,
                        top_n=1,  # Cukup ambil 1 teratas untuk evaluasi
                        min_conf=min_conf,
                        power=power,
                        mode='product',
                        window_dict=best_window_sizes,
                        mode_prediksi="hybrid"
                    )

                    if top_komb:
                        score = top_komb[0][1]
                        if score > best_score:
                            best_score = score
                            best_params.update({
                                "temperature": temp,
                                "confidence_power": power,
                                "min_confidence": min_conf,
                                "best_score": score
                            })
                except Exception as e:
                    # Abaikan error dan lanjutkan ke kombinasi berikutnya
                    continue
    
    progress_callback(1.0, "Analisa Selesai!")
    
    # Parameter lain yang diminta tetapi tidak ada di logika saat ini
    # Kita berikan nilai default atau rekomendasi umum
    best_params.update({
        "lstm_weight": 0.6,  # Nilai umum untuk ensemble
        "catboost_weight": 0.4, # Nilai umum untuk ensemble
        "heatmap_weight": 1.0, # Tidak relevan untuk model, lebih ke visualisasi
        "min_confidence_lstm": best_params.get("min_confidence", 0.001) # Samakan dengan min_confidence
    })


    return best_params
