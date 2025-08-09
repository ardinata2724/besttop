# Di dalam file ai_model.py, ganti fungsi ini:

def find_best_window_size_with_model_true(df, label, lokasi, model_type="lstm", min_ws=4, max_ws=20,
                                          temperature=1.0, use_cv=False, cv_folds=5, seed=42,
                                          min_acc=0.60, min_conf=0.60, top_n=6):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    best_ws = None
    best_score = -1

    table_data = []
    all_scores = []
    
    # --- Perubahan: Tidak lagi menampilkan header di sini ---
    # st.markdown(f"### 🔍 Pencarian WS - {label.upper()} (Top-{top_n})")
    status_placeholder = st.empty() # Placeholder untuk status
    ws_range = list(range(min_ws, max_ws + 1))

    for idx, ws in enumerate(ws_range):
        try:
            status_placeholder.info(f"🧠 Mencoba WS={ws} ({idx+1}/{len(ws_range)}) untuk **{label.upper()}**...")
            X, y_dict = preprocess_data(df, window_size=ws)
            if label not in y_dict or y_dict[label].shape[0] < 10:
                continue

            y = y_dict[label]
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
            model = build_transformer_model(X.shape[1]) if model_type == "transformer" else build_lstm_model(X.shape[1])
            model.compile(optimizer="adam",
                          loss="categorical_crossentropy",
                          metrics=["accuracy", TopKCategoricalAccuracy(k=top_n)])
            
            model.fit(
                X_train, y_train,
                epochs=15, batch_size=32, verbose=0,
                validation_data=(X_val, y_val),
                callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
            )
            
            eval_result = model.evaluate(X_val, y_val, verbose=0)
            val_acc = eval_result[1]
            top_n_acc = eval_result[2]

            preds = model.predict(X_val, verbose=0)
            avg_conf = np.mean(np.sort(preds, axis=1)[:, -top_n:])

            if val_acc < min_acc or avg_conf < min_conf:
                continue

            score = (val_acc * 0.35) + (top_n_acc * 0.35) + (avg_conf * 0.30)
            
            last_pred = model.predict(X[-1:], verbose=0)[0]
            top_n_digits_pred = np.argsort(last_pred)[::-1][:top_n]
            
            table_data.append((ws, round(val_acc*100, 2), round(top_n_acc*100, 2), round(avg_conf*100, 2), ", ".join(map(str, top_n_digits_pred))))
            all_scores.append((ws, val_acc, top_n_acc, avg_conf, top_n_digits_pred, score))

            if score > best_score:
                best_score = score
                best_ws = ws

        except Exception as e:
            st.warning(f"[GAGAL WS={ws}] {e}")
            continue
    
    status_placeholder.empty() # Hapus pesan status setelah selesai
    
    # --- Perubahan: Mengembalikan (return) hasil, bukan menampilkannya ---
    if not table_data:
        return None, None

    df_table = pd.DataFrame(table_data, columns=["Window Size", "Acc (%)", f"Top-{top_n} Acc (%)", "Conf (%)", f"Top-{top_n}"])
    
    return best_ws, df_table
