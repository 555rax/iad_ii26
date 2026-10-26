# -*- coding: utf-8 -*-
"""
Лабораторная работа № 4.
Предобучение нейронных сетей с использованием RBM.
Вариант 15: Cardiotocography (NSP). Доп. задание: Optical Digits (ЛР2).
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключает лишние предупреждения TensorFlow

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs('results', exist_ok=True)


# ============================================================
# ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ
# ============================================================
def load_cardio():
    """Данные для ЛР4 (Вариант 15) - Cardiotocography"""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=193)
    X = ds.data.features.copy()
    y = ds.data.targets['NSP'].values

    if 'CLASS' in X.columns:
        X = X.drop(columns=['CLASS'])
    if 'NSP' in X.columns:
        X = X.drop(columns=['NSP'])
    return X, y, "Cardiotocography (ЛР4)"


def load_optdigits():
    """Данные для ЛР2 - Optical recognition of handwritten digits"""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=80)
    X = ds.data.features.copy()
    y = ds.data.targets.iloc[:, 0].values
    return X, y, "Optical Digits (ЛР2)"


# ============================================================
# УТИЛИТЫ
# ============================================================
def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def evaluate(model, X_te, y_te):
    pred = model.predict(X_te, verbose=0)
    yp = np.argmax(pred, axis=1)
    return {
        'accuracy': float(np.mean(yp == y_te)),
        'f1_macro': float(f1_score(y_te, yp, average='macro')),
        'f1_weighted': float(f1_score(y_te, yp, average='weighted')),
        'cm': confusion_matrix(y_te, yp),
    }


# ============================================================
# ОСНОВНОЙ ПАЙПЛАЙН ЭКСПЕРИМЕНТОВ
# ============================================================
def run_experiment(X_raw, y_raw, dataset_name):
    print("\n" + "#" * 75)
    print(f"ЗАПУСК ЭКСПЕРИМЕНТОВ ДЛЯ: {dataset_name}")
    print("#" * 75)

    # ⚡ Кодируем классы в 0..N-1 (необходимо для Keras)
    classes = np.sort(np.unique(y_raw))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([class_to_idx[c] for c in y_raw], dtype=np.int32)
    print(f"Исходные классы: {classes} -> индексы 0..{len(classes) - 1}")

    # Разделение данных
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X_raw, y_encoded, test_size=0.30, random_state=SEED, stratify=y_encoded)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

    # Масштабирование
    scaler_std = StandardScaler()
    X_train = scaler_std.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler_std.transform(X_val_raw).astype(np.float32)
    X_test = scaler_std.transform(X_test_raw).astype(np.float32)

    # Для RBM нужен диапазон [0, 1]
    scaler_minmax = MinMaxScaler()
    X_train_rbm = scaler_minmax.fit_transform(X_train_raw).astype(np.float32)

    input_dim = X_train.shape[1]
    num_classes = len(classes)

    # Архитектура из ЛР3
    LAYER_DIMS = [128, 64, 32, 16]
    DROPOUT = 0.3
    EPOCHS = 80
    BATCH_SIZE = 32

    es = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # --------------------------------------------------------
    # 1. БАЗОВАЯ МОДЕЛЬ (Без предобучения)
    # --------------------------------------------------------
    print(f"\n[{dataset_name}] 1. Базовая модель...")
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for dim in LAYER_DIMS:
        x = layers.Dense(dim, activation='relu')(x)
        x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    base_model = compile_model(models.Model(inputs, outputs))
    base_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                   epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=0)
    res_base = evaluate(base_model, X_test, y_test)

    # --------------------------------------------------------
    # 2. АВТОЭНКОДЕР (ЛР3)
    # --------------------------------------------------------
    print(f"[{dataset_name}] 2. Предобучение AE...")
    current_train, current_val = X_train.copy(), X_val.copy()
    encoders = []
    for dim in LAYER_DIMS:
        inp = layers.Input(shape=(current_train.shape[1],))
        enc = layers.Dense(dim, activation='relu', name='encoder')(inp)
        dec = layers.Dense(current_train.shape[1], activation='linear')(enc)
        ae = models.Model(inp, dec)
        ae.compile(optimizer='adam', loss='mse')
        ae.fit(current_train, current_train, validation_data=(current_val, current_val),
               epochs=30, batch_size=BATCH_SIZE, verbose=0)

        encoder = models.Model(inp, enc)
        encoders.append(encoder)
        current_train = encoder.predict(current_train, verbose=0)
        current_val = encoder.predict(current_val, verbose=0)

    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for dim, enc in zip(LAYER_DIMS, encoders):
        dense = layers.Dense(dim, activation='relu')
        x = dense(x)
        dense.set_weights(enc.get_layer('encoder').get_weights())
        x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    ae_model = compile_model(models.Model(inputs, outputs))
    ae_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                 epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=0)
    res_ae = evaluate(ae_model, X_test, y_test)

    # --------------------------------------------------------
    # 3. RBM (ЛР4)
    # --------------------------------------------------------
    print(f"[{dataset_name}] 3. Предобучение RBM...")
    current_train_rbm = X_train_rbm.copy()
    rbm_weights = []
    for dim in LAYER_DIMS:
        # n_iter=20 - Условие останова для RBM
        rbm = BernoulliRBM(n_components=dim, learning_rate=0.01, n_iter=20,
                           batch_size=BATCH_SIZE, random_state=SEED, verbose=0)
        rbm.fit(current_train_rbm)
        W = rbm.components_.T
        b = rbm.intercept_hidden_
        rbm_weights.append((W, b))
        current_train_rbm = rbm.transform(current_train_rbm)

    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for dim, (W, b) in zip(LAYER_DIMS, rbm_weights):
        # ⚡ ВАЖНО: Используем sigmoid, так как RBM обучался на сигмоиде
        dense = layers.Dense(dim, activation='sigmoid')
        x = dense(x)
        dense.set_weights([W, b])
        x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    rbm_model = compile_model(models.Model(inputs, outputs))
    rbm_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=0)
    res_rbm = evaluate(rbm_model, X_test, y_test)

    # --------------------------------------------------------
    # ВЫВОД РЕЗУЛЬТАТОВ
    # --------------------------------------------------------
    print(f"\n--- РЕЗУЛЬТАТЫ ДЛЯ {dataset_name} ---")
    print(f"{'Модель':<30} {'Accuracy':<10} {'F1_macro':<10} {'F1_weighted':<10}")
    print("-" * 65)
    print(
        f"{'1. Без предобучения':<30} {res_base['accuracy']:<10.4f} {res_base['f1_macro']:<10.4f} {res_base['f1_weighted']:<10.4f}")
    print(
        f"{'2. С AE предобучением':<30} {res_ae['accuracy']:<10.4f} {res_ae['f1_macro']:<10.4f} {res_ae['f1_weighted']:<10.4f}")
    print(
        f"{'3. С RBM предобучением':<30} {res_rbm['accuracy']:<10.4f} {res_rbm['f1_macro']:<10.4f} {res_rbm['f1_weighted']:<10.4f}")
    print("-" * 65)

    # Построение матриц ошибок
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, res, title, cmap in zip(axes, [res_base, res_ae, res_rbm],
                                    ['Базовая', 'AE', 'RBM'],
                                    ['Blues', 'Greens', 'Oranges']):
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap=cmap, ax=ax)
        ax.set_title(f'{dataset_name} — {title}')
        ax.set_xlabel('Предсказано')
        ax.set_ylabel('Истина')
    plt.tight_layout()
    filename = f"results/cm_{dataset_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename, dpi=120)
    plt.close()


# ============================================================
# ЗАПУСК
# ============================================================
if __name__ == "__main__":
    # 1. Данные варианта 15 (Cardiotocography)
    try:
        X_cardio, y_cardio, name_cardio = load_cardio()
        run_experiment(X_cardio, y_cardio, name_cardio)
    except Exception as e:
        print(f"Ошибка Cardio: {e}")

    # 2. Данные ЛР2 (Optical Digits) - Пункт 3 задания
    try:
        X_opt, y_opt, name_opt = load_optdigits()
        run_experiment(X_opt, y_opt, name_opt)
    except Exception as e:
        print(f"Ошибка Optical Digits: {e}")