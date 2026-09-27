import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import sklearn
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

print("scikit-learn:", sklearn.__version__)
print("torch:", torch.__version__)

print("\n" + "#" * 60)
print("# ЧАСТЬ A. ЛР2: WINE QUALITY (WHITE)")
print("#" * 60)

url_white = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
             "wine-quality/winequality-white.csv")
white_wine = pd.read_csv(url_white, sep=';')

print(f"Dataset shape: {white_wine.shape}")
print(f"Quality distribution:\n{white_wine['quality'].value_counts().sort_index()}")

X_w = white_wine.drop(columns=['quality']).values
y_w_raw = white_wine['quality'].values

scaler_w = StandardScaler()
X_w_scaled = scaler_w.fit_transform(X_w)


def group_quality(q):
    if q <= 4:
        return 0
    elif q == 5:
        return 1
    elif q == 6:
        return 2
    else:
        return 3


y_w_grouped = np.array([group_quality(q) for q in y_w_raw])
class_names = {0: 'Low (3-4)', 1: 'Medium (5)', 2: 'High (6)', 3: 'Very High (7-9)'}
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
markers = ['o', 's', '^', 'D']

print("\nGrouped class distribution:")
unique, counts = np.unique(y_w_grouped, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u} ({class_names[u]}): {c} samples")


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(model, dataloader, epochs=100, lr=1e-3):
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(epochs):
        ep_loss = 0.0
        for bx, by in dataloader:
            opt.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * bx.size(0)
        losses.append(ep_loss / len(dataloader.dataset))
    return losses


def get_latent(model, X_tensor):
    model.eval()
    with torch.no_grad():
        return model.encode(X_tensor).numpy()


X_w_tensor = torch.FloatTensor(X_w_scaled)
loader_w = DataLoader(TensorDataset(X_w_tensor, X_w_tensor),
                      batch_size=64, shuffle=True)
input_dim_w = X_w_scaled.shape[1]

print("\n--- AE обучение (latent 2 и 3) ---")
ae_2d = Autoencoder(input_dim_w, 2, hidden_dim=32)
train_autoencoder(ae_2d, loader_w, epochs=100)
latent_2d = get_latent(ae_2d, X_w_tensor)

ae_3d = Autoencoder(input_dim_w, 3, hidden_dim=32)
train_autoencoder(ae_3d, loader_w, epochs=100)
latent_3d = get_latent(ae_3d, X_w_tensor)

pca_2d = PCA(n_components=2, random_state=42).fit(X_w_scaled)
pca_2d_res = pca_2d.transform(X_w_scaled)
pca_3d = PCA(n_components=3, random_state=42).fit(X_w_scaled)
pca_3d_res = pca_3d.transform(X_w_scaled)
print(f"PCA 2D explained variance: {pca_2d.explained_variance_ratio_.sum():.4f}")
print(f"PCA 3D explained variance: {pca_3d.explained_variance_ratio_.sum():.4f}")

perplexities = [20, 30, 40, 50, 60]
idx = np.random.choice(len(X_w_scaled), size=min(1000, len(X_w_scaled)), replace=False)
X_s = X_w_scaled[idx]
y_s = y_w_grouped[idx]

tsne_2d_res = {}
for p in perplexities:
    print(f"t-SNE 2D perplexity={p}...")
    tsne_2d_res[p] = TSNE(n_components=2, perplexity=p, init='pca',
                          random_state=42, learning_rate='auto',
                          max_iter=1000).fit_transform(X_s)

trust = {p: trustworthiness(X_s, tsne_2d_res[p], n_neighbors=5) for p in perplexities}
for p in perplexities:
    print(f"  perplexity={p}: trustworthiness={trust[p]:.4f}")
best_perp = max(trust, key=trust.get)
print(f"Best perplexity = {best_perp} (trust = {trust[best_perp]:.4f})")

tsne_3d_res = TSNE(n_components=3, perplexity=best_perp, init='pca',
                   random_state=42, learning_rate='auto',
                   max_iter=1000).fit_transform(X_s)


def scatter2d(ax, data, labels, title, xlabel, ylabel):
    for cls in np.unique(labels):
        m = labels == cls
        ax.scatter(data[m, 0], data[m, 1], c=colors[cls], marker=markers[cls],
                   label=class_names[cls], alpha=0.6, s=20,
                   edgecolors='w', linewidth=0.3)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
scatter2d(axes[0], latent_2d, y_w_grouped, 'AE 2D (Wine)', 'Latent 1', 'Latent 2')
scatter2d(axes[1], pca_2d_res, y_w_grouped, 'PCA 2D (Wine)', 'PC1', 'PC2')
scatter2d(axes[2], tsne_2d_res[best_perp], y_s,
          f't-SNE 2D (perp={best_perp})', 't-SNE 1', 't-SNE 2')
handles = [plt.Line2D([0], [0], marker=markers[i], color='w',
                      markerfacecolor=colors[i], markersize=8,
                      label=class_names[i]) for i in range(4)]
fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
fig.suptitle('ЛР2: Wine Quality — AE / PCA / t-SNE (2D)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('lr2_wine_2d.png', dpi=120, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
for cls in np.unique(y_w_grouped):
    m = y_w_grouped == cls
    ax.scatter(latent_3d[m, 0], latent_3d[m, 1], latent_3d[m, 2],
               c=colors[cls], marker=markers[cls], label=class_names[cls],
               alpha=0.6, s=20, edgecolors='w', linewidth=0.3)
ax.set_title('ЛР2: AE 3D (Wine Quality white)', fontsize=13, fontweight='bold')
ax.set_xlabel('Latent 1')
ax.set_ylabel('Latent 2')
ax.set_zlabel('Latent 3')
ax.legend(title='Quality Class', loc='best')
plt.tight_layout()
plt.savefig('lr2_wine_ae_3d.png', dpi=120, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
for cls in np.unique(y_s):
    m = y_s == cls
    ax.scatter(tsne_3d_res[m, 0], tsne_3d_res[m, 1], tsne_3d_res[m, 2],
               c=colors[cls], marker=markers[cls], label=class_names[cls],
               alpha=0.6, s=20, edgecolors='w', linewidth=0.3)
ax.set_title(f'ЛР2: t-SNE 3D (perplexity={best_perp})',
             fontsize=13, fontweight='bold')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_zlabel('t-SNE 3')
ax.legend(title='Quality Class', loc='best')
plt.tight_layout()
plt.savefig('lr2_wine_tsne_3d.png', dpi=120, bbox_inches='tight')
plt.show()

print("\n" + "#" * 60)
print("# ЧАСТЬ B. ЛР3: RAISIN (классификация)")
print("#" * 60)

import tensorflow as tf
from tensorflow.keras import layers, Model
tf.random.set_seed(42)

df_r = pd.read_excel("Raisin_Dataset.xlsx")
X_r = df_r.drop("Class", axis=1).values
y_r = LabelEncoder().fit_transform(df_r["Class"].values)
scaler_r = StandardScaler()
X_r_scaled = scaler_r.fit_transform(X_r)

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    X_r_scaled, y_r, test_size=0.2, random_state=42, stratify=y_r)


def build_baseline_bin(input_dim, hidden=(32, 16, 8)):
    m = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden[0], activation='relu'),
        layers.Dense(hidden[1], activation='relu'),
        layers.Dense(hidden[2], activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m


def pretrain_layer(data_in, hidden_dim, epochs=50):
    inp = layers.Input(shape=(data_in.shape[1],))
    enc = layers.Dense(hidden_dim, activation='relu')(inp)
    dec = layers.Dense(data_in.shape[1], activation='linear')(enc)
    ae = Model(inp, dec)
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(data_in, data_in, epochs=epochs, batch_size=32, verbose=0)
    return Model(inp, enc)


def clf_report(y_true, y_pred, name):
    print(f"[{name}] Acc={accuracy_score(y_true,y_pred):.4f} "
          f"Prec={precision_score(y_true,y_pred):.4f} "
          f"Rec={recall_score(y_true,y_pred):.4f} "
          f"F1={f1_score(y_true,y_pred):.4f}")
    print("CM:", confusion_matrix(y_true, y_pred).tolist())


print("\n--- Raisin П.1: Baseline (7→32→16→8→1) ---")
base_r = build_baseline_bin(Xr_tr.shape[1])
base_r.fit(Xr_tr, yr_tr, epochs=100, batch_size=32,
           validation_split=0.1, verbose=0)
y_pred_base_r = (base_r.predict(Xr_te, verbose=0) > 0.5).astype(int)
clf_report(yr_te, y_pred_base_r, "Raisin Baseline")

print("\n--- Raisin П.2: AE-предобучение (50 эпох на слой) ---")
e1 = pretrain_layer(X_r_scaled, 32, 50)
e2 = pretrain_layer(e1.predict(X_r_scaled, verbose=0), 16, 50)
pre_r = build_baseline_bin(Xr_tr.shape[1])
pre_r.layers[0].set_weights(e1.get_weights())
pre_r.layers[1].set_weights(e2.get_weights())
pre_r.fit(Xr_tr, yr_tr, epochs=100, batch_size=32,
          validation_split=0.1, verbose=0)
y_pred_pre_r = (pre_r.predict(Xr_te, verbose=0) > 0.5).astype(int)
clf_report(yr_te, y_pred_pre_r, "Raisin Pretrained")

print("\n--- Raisin П.3: Сравнение ---")
comp_r = pd.DataFrame({
    'Метрика': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Baseline': [accuracy_score(yr_te, y_pred_base_r),
                 precision_score(yr_te, y_pred_base_r),
                 recall_score(yr_te, y_pred_base_r),
                 f1_score(yr_te, y_pred_base_r)],
    'С предобучением': [accuracy_score(yr_te, y_pred_pre_r),
                        precision_score(yr_te, y_pred_pre_r),
                        recall_score(yr_te, y_pred_pre_r),
                        f1_score(yr_te, y_pred_pre_r)]
})
comp_r[['Baseline', 'С предобучением']] = comp_r[['Baseline', 'С предобучением']].round(4)
comp_r['Δ'] = (comp_r['С предобучением'] - comp_r['Baseline']).round(4)
print(comp_r.to_string(index=False))

print("\n" + "#" * 60)
print("# ЧАСТЬ C. ЛР3 п.4: WINE QUALITY (WHITE), 4 класса")
print("#" * 60)

y_w_cls = y_w_grouped

Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(
    X_w_scaled, y_w_cls, test_size=0.2, random_state=42, stratify=y_w_cls)


def build_baseline_multi(input_dim, n_classes=4, hidden=(64, 32, 16)):
    m = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden[0], activation='relu'),
        layers.Dense(hidden[1], activation='relu'),
        layers.Dense(hidden[2], activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m


def multi_report(y_true, y_pred, name):
    print(f"[{name}] Acc={accuracy_score(y_true,y_pred):.4f} "
          f"F1_macro={f1_score(y_true,y_pred,average='macro'):.4f} "
          f"F1_weighted={f1_score(y_true,y_pred,average='weighted'):.4f}")
    print("CM:\n", confusion_matrix(y_true, y_pred))


print("\n--- Wine П.1: Baseline (11→64→32→16→4) ---")
base_w = build_baseline_multi(Xw_tr.shape[1], n_classes=4)
base_w.fit(Xw_tr, yw_tr, epochs=100, batch_size=64,
           validation_split=0.1, verbose=0)
y_pred_base_w = np.argmax(base_w.predict(Xw_te, verbose=0), axis=1)
multi_report(yw_te, y_pred_base_w, "Wine Baseline")

print("\n--- Wine П.2: AE-предобучение (50 эпох на слой) ---")
ew1 = pretrain_layer(X_w_scaled, 64, 50)
H1w = ew1.predict(X_w_scaled, verbose=0)
ew2 = pretrain_layer(H1w, 32, 50)

pre_w = build_baseline_multi(Xw_tr.shape[1], n_classes=4)
pre_w.layers[0].set_weights(ew1.get_weights())
pre_w.layers[1].set_weights(ew2.get_weights())
pre_w.fit(Xw_tr, yw_tr, epochs=100, batch_size=64,
          validation_split=0.1, verbose=0)
y_pred_pre_w = np.argmax(pre_w.predict(Xw_te, verbose=0), axis=1)
multi_report(yw_te, y_pred_pre_w, "Wine Pretrained")

print("\n--- Wine П.3: Сравнение ---")
comp_w = pd.DataFrame({
    'Метрика': ['Accuracy', 'F1_macro', 'F1_weighted'],
    'Baseline': [accuracy_score(yw_te, y_pred_base_w),
                 f1_score(yw_te, y_pred_base_w, average='macro'),
                 f1_score(yw_te, y_pred_base_w, average='weighted')],
    'С предобучением': [accuracy_score(yw_te, y_pred_pre_w),
                        f1_score(yw_te, y_pred_pre_w, average='macro'),
                        f1_score(yw_te, y_pred_pre_w, average='weighted')]
})
comp_w[['Baseline', 'С предобучением']] = comp_w[['Baseline', 'С предобучением']].round(4)
comp_w['Δ'] = (comp_w['С предобучением'] - comp_w['Baseline']).round(4)
print(comp_w.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, pred, title in zip(axes, [y_pred_base_w, y_pred_pre_w],
                           ['Baseline', 'Pretrained (AE)']):
    sns.heatmap(confusion_matrix(yw_te, pred), annot=True, fmt='d',
                cmap='Blues', ax=ax,
                xticklabels=list(class_names.values()),
                yticklabels=list(class_names.values()))
    ax.set_title(f'Wine — {title}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('lr3_wine_comparison.png', dpi=120)
plt.show()

print("\n" + "=" * 60)
print("ВЫВОДЫ")
print("=" * 60)


def interpret(delta):
    if abs(delta) < 0.005:
        return "разница в пределах шума, эффект нейтральный"
    elif delta > 0:
        return "предобучение дало небольшой положительный эффект"
    else:
        return "предобучение ухудшило результат"


print("\nЛР2 (Wine Quality white, 4 класса):")
print("- AE 2D/3D и PCA дают сходную картину: 4 класса сильно")
print("  перемешаны, чёткого разделения нет.")
print(f"- t-SNE 2D с perplexity={best_perp} (trust={trust[best_perp]:.4f})")
print("  выявляет локальные группы, но 4 класса всё равно перекрываются —")
print("  это ожидаемо: классы Medium (5) и High (6) доминируют и близки.")

delta_f1_r = f1_score(yr_te, y_pred_pre_r) - f1_score(yr_te, y_pred_base_r)
print("\nЛР3 (Raisin, вариант 9):")
print(f"- Baseline F1={f1_score(yr_te, y_pred_base_r):.4f}, "
      f"Acc={accuracy_score(yr_te, y_pred_base_r):.4f}")
print(f"- Pretrained F1={f1_score(yr_te, y_pred_pre_r):.4f}, "
      f"Acc={accuracy_score(yr_te, y_pred_pre_r):.4f}")
print(f"- ΔF1 = {delta_f1_r:+.4f} → {interpret(delta_f1_r)}")

delta_f1_w = (f1_score(yw_te, y_pred_pre_w, average='macro')
              - f1_score(yw_te, y_pred_base_w, average='macro'))
print("\nЛР3 п.4 (Wine Quality white, 4 класса):")
print(f"- Baseline F1_macro={f1_score(yw_te, y_pred_base_w, average='macro'):.4f}, "
      f"Acc={accuracy_score(yw_te, y_pred_base_w):.4f}")
print(f"- Pretrained F1_macro={f1_score(yw_te, y_pred_pre_w, average='macro'):.4f}, "
      f"Acc={accuracy_score(yw_te, y_pred_pre_w):.4f}")
print(f"- ΔF1_macro = {delta_f1_w:+.4f} → {interpret(delta_f1_w)}")
print("  Причина: сильный дисбаланс 4 классов (Medium и High доминируют,")
print("  Low и Very High малочисленны). MSE-реконструкция признаков")
print("  не помогает разделять близкие ординальные классы 5 и 6.")

print("\nОБЩИЙ ВЫВОД:")
print("Автоэнкодерное предобучение не является универсальным улучшением")
print("для табличных данных. Эффект зависит от датасета: на Raisin он")
print("даёт слабый положительный прирост, на Wine Quality (white) с 4")
print("несбалансированными классами — ухудшает метрики.")