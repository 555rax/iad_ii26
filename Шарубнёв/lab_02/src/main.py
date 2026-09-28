import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


np.random.seed(42)
torch.manual_seed(42)


DATA_PATH = "winequality-white.csv"

if not os.path.exists(DATA_PATH):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, sep=';')
else:
    df = pd.read_csv(DATA_PATH, sep=';')

print("--- Информация о датасете ---")
print(f"Размерность: {df.shape}")
print(f"Пропуски:\n{df.isna().sum().sum()}")
print(f"Распределение целевого класса (quality):\n{df['quality'].value_counts().sort_index()}")

X = df.drop(columns=["quality"])
y = df["quality"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

classes = np.unique(y)
num_classes = len(classes)


def plot_2d_scatter(Z, title, y_labels):
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y_labels, cmap='viridis', alpha=0.6, edgecolors='none', s=25)
    cbar = plt.colorbar(scatter, ticks=np.unique(y_labels))
    cbar.set_label('Wine Quality')
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_3d_scatter(Z, title, y_labels):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=y_labels, cmap='viridis', alpha=0.6, s=20)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, ticks=np.unique(y_labels))
    cbar.set_label('Wine Quality')
    ax.set_xlabel("Comp 1")
    ax.set_ylabel("Comp 2")
    ax.set_zlabel("Comp 3")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()



print("\n=== Выполнение PCA ===")

cov_matrix = np.cov(X_scaled, rowvar=False)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
eig_vals = np.real(eig_vals)
eig_vecs = np.real(eig_vecs)

idx = np.argsort(eig_vals)[::-1]
eig_vals_sorted = eig_vals[idx]
eig_vecs_sorted = eig_vecs[:, idx]

Z2_pca_manual = X_scaled @ eig_vecs_sorted[:, :2]
Z3_pca_manual = X_scaled @ eig_vecs_sorted[:, :3]

pca2 = PCA(n_components=2)
Z2_pca_sk = pca2.fit_transform(X_scaled)

pca3 = PCA(n_components=3)
Z3_pca_sk = pca3.fit_transform(X_scaled)

total_var = np.sum(eig_vals_sorted)
var_exp_2 = np.sum(eig_vals_sorted[:2]) / total_var
var_exp_3 = np.sum(eig_vals_sorted[:3]) / total_var

print(f"PCA (2 компоненты) объясняет {var_exp_2:.4f} дисперсии")
print(f"PCA (3 компоненты) объясняет {var_exp_3:.4f} дисперсии")

plot_2d_scatter(Z2_pca_sk, "PCA 2D Projection (sklearn)", y)
plot_3d_scatter(Z3_pca_sk, "PCA 3D Projection (sklearn)", y)


print("\n=== Обучение Автоэнкодеров ===")


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Кодировщик (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(X_data, latent_dim, epochs=120, batch_size=64, lr=0.001):
    input_dim = X_data.shape[1]
    model = Autoencoder(input_dim, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.FloatTensor(X_data))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(dataloader.dataset)
        if (epoch + 1) % 40 == 0 or epoch == epochs - 1:
            print(f"Latent Dim {latent_dim} | Epoch [{epoch + 1}/{epochs}] - Loss: {train_loss:.6f}")

    model.eval()
    with torch.no_grad():
        latent_space = model.encode(torch.FloatTensor(X_data)).numpy()
    return latent_space


Z2_ae = train_autoencoder(X_scaled, latent_dim=2)
Z3_ae = train_autoencoder(X_scaled, latent_dim=3)

plot_2d_scatter(Z2_ae, "Autoencoder 2D Bottleneck Representation", y)
plot_3d_scatter(Z3_ae, "Autoencoder 3D Bottleneck Representation", y)

print("\n=== Выполнение t-SNE ===")

perplexities = [20, 30, 45, 60]
best_perp = 45

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, perp in enumerate(perplexities):
    tsne_temp = TSNE(n_components=2, perplexity=perp, init='pca', learning_rate='auto', random_state=42)
    Z2_tsne_temp = tsne_temp.fit_transform(X_scaled)

    sc = axes[i].scatter(Z2_tsne_temp[:, 0], Z2_tsne_temp[:, 1], c=y, cmap='viridis', alpha=0.6, s=15)
    axes[i].set_title(f"t-SNE 2D (Perplexity = {perp})")
    axes[i].grid(True, linestyle='--', alpha=0.5)

fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label='Wine Quality')
plt.suptitle("Сравнение t-SNE 2D при различных значениях Perplexity", fontsize=14)
plt.show()

print(f"Построение итогового t-SNE (2D и 3D) с perplexity={best_perp}...")
tsne_2d = TSNE(n_components=2, perplexity=best_perp, init='pca', learning_rate='auto', random_state=42)
Z2_tsne = tsne_2d.fit_transform(X_scaled)

tsne_3d = TSNE(n_components=3, perplexity=best_perp, init='pca', learning_rate='auto', random_state=42)
Z3_tsne = tsne_3d.fit_transform(X_scaled)

plot_2d_scatter(Z2_tsne, f"t-SNE 2D Projection (Perplexity={best_perp})", y)
plot_3d_scatter(Z3_tsne, f"t-SNE 3D Projection (Perplexity={best_perp})", y)