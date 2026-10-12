# -*- coding: utf-8 -*-
"""
Запуск:
    python main.py infrared   # Infrared Thermography (aveOralF) — регрессия
    python main.py wine       # Wine Quality (white) — классификация
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report,
    mean_absolute_percentage_error, mean_squared_error, r2_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


DATASET = "infrared"      # "wine" | "infrared"
if len(sys.argv) > 1:
    DATASET = sys.argv[1].lower()

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs('results', exist_ok=True)



def load_wine():
    url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/"
           "wine-quality/winequality-white.csv")
    df = pd.read_csv(url, delimiter=';')
    return {
        'name': 'Wine Quality (white)',
        'task': 'classification',
        'X': df.drop('quality', axis=1),
        'y': df['quality'].values,
        'cat_cols': [],
        'batch_size': 64,
        'epochs': 200,
        'ae_epochs': 50,
    }


def load_infrared():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=925)
    X = ds.data.features.copy()
    y = ds.data.targets['aveOralF'].values
    if 'SubjectID' in X.columns:
        X = X.drop(columns=['SubjectID'])
    cat_cols = [c for c in ['Gender', 'Age', 'Ethnicity'] if c in X.columns]
    return {
        'name': 'Infrared Thermography (aveOralF)',
        'task': 'regression',
        'X': X, 'y': y.astype(np.float32),
        'cat_cols': cat_cols,
        'batch_size': 32,
        'epochs': 300,
        'ae_epochs': 50,
    }


LOADERS = {'wine': load_wine, 'infrared': load_infrared}
cfg = LOADERS[DATASET]()
print("=" * 70)
print(f"Датасет: {cfg['name']}  |  Задача: {cfg['task']}")
print("=" * 70)


X_raw, y_raw = cfg['X'], cfg['y']

print(f"Пропусков в X: {int(X_raw.isnull().sum().sum())}")
print(f"Пропусков в y: {int(pd.isnull(y_raw).sum())}")

cat_cols = cfg['cat_cols']
num_cols = [c for c in X_raw.columns if c not in cat_cols]

# ⚡ Числовые: медианная импутация + стандартизация
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

transformers = [('num', num_pipe, num_cols)]


if cat_cols:
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(drop='first', sparse_output=False,
                                  handle_unknown='ignore')),
    ])
    transformers.append(('cat', cat_pipe, cat_cols))

preprocessor = ColumnTransformer(transformers)
X = preprocessor.fit_transform(X_raw).astype(np.float32)

if np.isnan(X).any():
    print(f"⚠ Осталось NaN: {int(np.isnan(X).sum())} → заменяем на 0")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


if cfg['task'] == 'classification':

    from collections import Counter
    counts = Counter(y_raw)
    MIN_COUNT = 30
    sorted_vals = sorted(counts.keys())
    merge_map = {}
    for v in sorted_vals:
        if counts[v] < MIN_COUNT:
            neighbors = [c for c in sorted_vals if counts[c] >= MIN_COUNT]
            if neighbors:
                nearest = min(neighbors, key=lambda c: abs(c - v))
                merge_map[v] = nearest

    if merge_map:
        print(f"Объединение редких классов (порог {MIN_COUNT}): {merge_map}")
        y_raw = np.array([merge_map.get(v, v) for v in y_raw])
        print(f"Новое распределение: {dict(Counter(y_raw))}")


    classes = np.sort(np.unique(y_raw))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in y_raw], dtype=np.int32)
    num_classes = len(classes)
    cfg['target_names'] = [str(c) for c in classes]
    cfg['y_mean'] = cfg['y_std'] = None
    print(f"Классы: {classes}, распределение: {np.bincount(y)}")
else:
    y = y_raw.astype(np.float32)
    y_mean, y_std = float(y.mean()), float(y.std())
    y = ((y - y_mean) / y_std).astype(np.float32)
    cfg['y_mean'], cfg['y_std'] = y_mean, y_std
    num_classes = 1
    print(f"y: mean={y_mean:.4f}, std={y_std:.4f}")

print(f"X: {X.shape}")

strat = y if cfg['task'] == 'classification' else None
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=strat)
strat2 = y_temp if cfg['task'] == 'classification' else None
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=strat2)
input_dim = X_train.shape[1]
print(f"Train/Val/Test: {X_train.shape} / {X_val.shape} / {X_test.shape}")



if cfg['task'] == 'regression':
    y_orig = y * y_std + y_mean
    feat_names = preprocessor.get_feature_names_out()
    corrs = []
    for i, name in enumerate(feat_names):
        try:
            r = abs(np.corrcoef(X[:, i], y_orig)[0, 1])
            if not np.isnan(r):
                corrs.append((name, r))
        except Exception:
            pass
    corrs.sort(key=lambda x: -x[1])
    print("\nТоп-10 корреляций с таргетом:")
    for name, r in corrs[:10]:
        print(f"  {name:<30} |r| = {r:.4f}")

    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    ridge_r2 = r2_score(y_test, ridge.predict(X_test))
    print(f"\nRidge baseline R² = {ridge_r2:.4f}")
    print("(для сравнения с нейросетевой моделью)\n")



def make_output_layer(task, num_classes):
    if task == 'classification':
        return layers.Dense(num_classes, activation='softmax', name='output')
    return layers.Dense(1, activation='linear', name='output')


def compile_model(model, task, lr):
    if task == 'classification':
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss='mse',
                      metrics=[tf.keras.metrics.MeanAbsolutePercentageError(name='mape')])
    return model


def evaluate(model, X_te, y_te, task, y_mean=None, y_std=None):
    pred = model.predict(X_te, verbose=0)
    if task == 'classification':
        yp = np.argmax(pred, axis=1)
        return {
            'y_pred': yp,
            'accuracy': float(np.mean(yp == y_te)),
            'f1_macro': float(f1_score(y_te, yp, average='macro')),
            'f1_weighted': float(f1_score(y_te, yp, average='weighted')),
            'cm': confusion_matrix(y_te, yp),
        }
    yp = pred.ravel() * y_std + y_mean
    yt = y_te * y_std + y_mean
    return {
        'y_pred': yp, 'y_true': yt,
        'mse':  float(mean_squared_error(yt, yp)),
        'mape': float(mean_absolute_percentage_error(yt, yp) * 100),
        'r2':   float(r2_score(yt, yp)),
    }



LAYER_DIMS = [128, 64, 32, 16]
DROPOUT = 0.3


def build_base_model(input_dim, task, num_classes):
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for i, dim in enumerate(LAYER_DIMS):
        x = layers.Dense(dim, activation='relu', name=f'hidden{i+1}')(x)
        x = layers.Dropout(DROPOUT)(x)
    x = make_output_layer(task, num_classes)(x)
    return compile_model(models.Model(inputs, x), task, lr=1e-3)



print("=" * 70); print("БАЗОВАЯ МОДЕЛЬ"); print("=" * 70)
base_model = build_base_model(input_dim, cfg['task'], num_classes)
base_model.summary()

es_base = callbacks.EarlyStopping(
    monitor='val_loss', patience=50, restore_best_weights=True)

hist_base = base_model.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=cfg['epochs'], batch_size=cfg['batch_size'],
    callbacks=[es_base], verbose=0)

res_base = evaluate(base_model, X_test, y_test, cfg['task'],
                    cfg.get('y_mean'), cfg.get('y_std'))
print(f"\nБазовая: "
      f"{ {k: round(v,4) for k,v in res_base.items() if isinstance(v,float)} }")



def build_autoencoder(input_dim, encoding_dim):
    inp = layers.Input(shape=(input_dim,), name='ae_input')
    enc = layers.Dense(encoding_dim, activation='relu', name='encoder')(inp)
    dec = layers.Dense(input_dim, activation='linear', name='decoder')(enc)
    ae = models.Model(inp, dec, name='autoencoder')
    encoder = models.Model(inp, enc, name='encoder_model')
    ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return ae, encoder


print("\n" + "=" * 70); print("АВТОЭНКОДЕРНОЕ ПРЕДОБУЧЕНИЕ"); print("=" * 70)
current_train, current_val = X_train.copy(), X_val.copy()
pretrained_encoders = []

for i, dim in enumerate(LAYER_DIMS):
    print(f"\n--- AE {i+1} (dim={dim}) ---")
    ae, encoder = build_autoencoder(current_train.shape[1], dim)
    es_ae = callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                     restore_best_weights=True)
    ae.fit(current_train, current_train,
           validation_data=(current_val, current_val),
           epochs=cfg['ae_epochs'], batch_size=cfg['batch_size'],
           callbacks=[es_ae], verbose=0)
    pretrained_encoders.append(encoder)
    current_train = encoder.predict(current_train, verbose=0)
    current_val   = encoder.predict(current_val,   verbose=0)
    print(f"    → {current_train.shape}")



def build_pretrained_model(input_dim, layer_dims, encoders,
                           task, num_classes, dropout=0.3):
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for i, (dim, enc) in enumerate(zip(layer_dims, encoders)):
        dense = layers.Dense(dim, activation='relu', name=f'hidden{i+1}')
        x = dense(x)
        dense.set_weights(enc.get_layer('encoder').get_weights())
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    x = make_output_layer(task, num_classes)(x)
    return compile_model(models.Model(inputs, x), task, lr=1e-3)


print("\n" + "=" * 70); print("МОДЕЛЬ С ПРЕДОБУЧЕНИЕМ"); print("=" * 70)
pre_model = build_pretrained_model(
    input_dim, LAYER_DIMS, pretrained_encoders, cfg['task'], num_classes)
pre_model.summary()

es_pre = callbacks.EarlyStopping(
    monitor='val_loss', patience=50, restore_best_weights=True)

hist_pre = pre_model.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=cfg['epochs'], batch_size=cfg['batch_size'],
    callbacks=[es_pre], verbose=0)

res_pre = evaluate(pre_model, X_test, y_test, cfg['task'],
                   cfg.get('y_mean'), cfg.get('y_std'))
print(f"\nПредобученная: "
      f"{ {k: round(v,4) for k,v in res_pre.items() if isinstance(v,float)} }")



print("\n" + "=" * 70); print("СРАВНЕНИЕ"); print("=" * 70)

if cfg['task'] == 'classification':
    print(f"{'Модель':<25} {'Acc':<10} {'F1_macro':<10} {'F1_weighted':<12}")
    print("-" * 60)
    print(f"{'Без предобучения':<25} {res_base['accuracy']:<10.4f} "
          f"{res_base['f1_macro']:<10.4f} {res_base['f1_weighted']:<12.4f}")
    print(f"{'С предобучением':<25} {res_pre['accuracy']:<10.4f} "
          f"{res_pre['f1_macro']:<10.4f} {res_pre['f1_weighted']:<12.4f}")
    print("-" * 60)
    print(f"Δ Accuracy:    {(res_pre['accuracy']-res_base['accuracy'])*100:+.2f} п.п.")
    print(f"Δ F1_macro:    {(res_pre['f1_macro']-res_base['f1_macro'])*100:+.2f} п.п.")
    print(f"Δ F1_weighted: {(res_pre['f1_weighted']-res_base['f1_weighted'])*100:+.2f} п.п.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(res_base['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=cfg['target_names'], yticklabels=cfg['target_names'], ax=axes[0])
    axes[0].set_title('CM — без предобучения')
    axes[0].set_xlabel('Предсказано'); axes[0].set_ylabel('Истина')
    sns.heatmap(res_pre['cm'], annot=True, fmt='d', cmap='Greens',
                xticklabels=cfg['target_names'], yticklabels=cfg['target_names'], ax=axes[1])
    axes[1].set_title('CM — с предобучением')
    axes[1].set_xlabel('Предсказано'); axes[1].set_ylabel('Истина')
    plt.tight_layout(); plt.savefig('results/confusion_matrices.png', dpi=120); plt.close()

    print("\nОтчёт без предобучения:")
    print(classification_report(y_test, res_base['y_pred'],
          target_names=cfg['target_names'], zero_division=0))
    print("Отчёт с предобучением:")
    print(classification_report(y_test, res_pre['y_pred'],
          target_names=cfg['target_names'], zero_division=0))

else:
    print(f"{'Модель':<25} {'MSE':<10} {'MAPE (%)':<10} {'R²':<10}")
    print("-" * 55)
    print(f"{'Без предобучения':<25} {res_base['mse']:<10.4f} "
          f"{res_base['mape']:<10.2f} {res_base['r2']:<10.4f}")
    print(f"{'С предобучением':<25} {res_pre['mse']:<10.4f} "
          f"{res_pre['mape']:<10.2f} {res_pre['r2']:<10.4f}")
    print("-" * 55)
    print(f"Δ MSE: {res_pre['mse']-res_base['mse']:+.4f}   "
          f"Δ MAPE: {res_pre['mape']-res_base['mape']:+.2f} п.п.   "
          f"Δ R²: {res_pre['r2']-res_base['r2']:+.4f}")

    yt = res_base['y_true']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lims = [min(yt.min(), res_base['y_pred'].min(), res_pre['y_pred'].min()),
            max(yt.max(), res_base['y_pred'].max(), res_pre['y_pred'].max())]
    axes[0].scatter(yt, res_base['y_pred'], alpha=0.5, s=14, color='tab:blue')
    axes[0].plot(lims, lims, 'r--')
    axes[0].set_title(f"Без предобучения (R²={res_base['r2']:.3f})")
    axes[0].set_xlabel('Истина'); axes[0].set_ylabel('Прогноз')
    axes[1].scatter(yt, res_pre['y_pred'], alpha=0.5, s=14, color='green')
    axes[1].plot(lims, lims, 'r--')
    axes[1].set_title(f"С предобучением (R²={res_pre['r2']:.3f})")
    axes[1].set_xlabel('Истина'); axes[1].set_ylabel('Прогноз')
    plt.tight_layout(); plt.savefig('results/predictions.png', dpi=120); plt.close()



fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, hist, title in [(axes[0], hist_base, 'Базовая'),
                        (axes[1], hist_pre,  'С предобучением')]:
    ax.plot(hist.history['loss'], label='train')
    ax.plot(hist.history['val_loss'], label='val')
    ax.set_title(f'{title} — loss'); ax.set_xlabel('Эпоха'); ax.legend()
plt.tight_layout(); plt.savefig('results/training_history.png', dpi=120); plt.close()

print("\nГрафики в results/  |  Готово.")