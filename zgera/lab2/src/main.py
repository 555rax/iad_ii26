import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim

train = pd.read_csv("optdigits.tra", header=None)
test = pd.read_csv("optdigits.tes", header=None)
df = pd.concat([train, test], ignore_index=True)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print("Размер данных:", X.shape)
print("Классы:", sorted(set(y)))
print("Пример строки:", X[0])

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
idx = np.argsort(eig_vals)[::-1]
eig_vals_sorted = eig_vals[idx]
eig_vecs_sorted = eig_vecs[:, idx]

W2 = eig_vecs_sorted[:, :2]
W3 = eig_vecs_sorted[:, :3]

X_pca2_manual = X_std @ W2
X_pca3_manual = X_std @ W3

explained_ratio = eig_vals_sorted / np.sum(eig_vals_sorted)
print("PCA ручной — доля объяснённой дисперсии (2):", explained_ratio[:2].sum())
print("PCA ручной — доля объяснённой дисперсии (3):", explained_ratio[:3].sum())

pca2 = PCA(n_components=2)
X_pca2_skl = pca2.fit_transform(X_std)

pca3 = PCA(n_components=3)
X_pca3_skl = pca3.fit_transform(X_std)

print("PCA sklearn — доля объяснённой дисперсии (2):", pca2.explained_variance_ratio_.sum())
print("PCA sklearn — доля объяснённой дисперсии (3):", pca3.explained_variance_ratio_.sum())

plt.figure(figsize=(7, 6))
plt.scatter(X_pca2_manual[:, 0], X_pca2_manual[:, 1], c=y, cmap="tab10", alpha=0.7)
plt.title("Ручной PCA — 2 компоненты")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 6))
plt.scatter(X_pca2_skl[:, 0], X_pca2_skl[:, 1], c=y, cmap="tab10", alpha=0.7)
plt.title("sklearn PCA — 2 компоненты")
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca3_manual[:, 0], X_pca3_manual[:, 1], X_pca3_manual[:, 2], c=y, cmap="tab10", alpha=0.7)
ax.set_title("Ручной PCA — 3 компоненты")
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca3_skl[:, 0], X_pca3_skl[:, 1], X_pca3_skl[:, 2], c=y, cmap="tab10", alpha=0.7)
ax.set_title("sklearn PCA — 3 компоненты")
plt.show()

X_tensor = torch.tensor(X_std, dtype=torch.float32)

class AE2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

ae2 = AE2(X_std.shape[1])
opt = optim.Adam(ae2.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(50):
    opt.zero_grad()
    out = ae2(X_tensor)
    loss = loss_fn(out, X_tensor)
    loss.backward()
    opt.step()

print("AE2 финальная ошибка:", loss.item())

with torch.no_grad():
    X_ae2 = ae2.encoder(X_tensor).numpy()

plt.figure(figsize=(7, 6))
plt.scatter(X_ae2[:, 0], X_ae2[:, 1], c=y, cmap="tab10", alpha=0.7)
plt.title("Автоэнкодер — 2 нейрона")
plt.grid(True)
plt.show()

class AE3(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

ae3 = AE3(X_std.shape[1])
opt = optim.Adam(ae3.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(50):
    opt.zero_grad()
    out = ae3(X_tensor)
    loss = loss_fn(out, X_tensor)
    loss.backward()
    opt.step()

print("AE3 финальная ошибка:", loss.item())

with torch.no_grad():
    X_ae3 = ae3.encoder(X_tensor).numpy()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_ae3[:, 0], X_ae3[:, 1], X_ae3[:, 2], c=y, cmap="tab10", alpha=0.7)
ax.set_title("Автоэнкодер — 3 нейрона")
plt.show()

tsne2 = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne2 = tsne2.fit_transform(X_std)

plt.figure(figsize=(7, 6))
plt.scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=y, cmap="tab10", alpha=0.7)
plt.title("t-SNE — 2 компоненты")
plt.grid(True)
plt.show()

tsne3 = TSNE(n_components=3, perplexity=30, random_state=42)
X_tsne3 = tsne3.fit_transform(X_std)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_tsne3[:, 0], X_tsne3[:, 1], X_tsne3[:, 2], c=y, cmap="tab10", alpha=0.7)
ax.set_title("t-SNE — 3 компоненты")
plt.show()
