import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

colors = {0: 'r', 1: 'g'}
labels_map = {0: 'malignant', 1: 'benign'}

def plot_2d(X2, title, filename):
    plt.figure(figsize=(7, 6))
    for c in np.unique(y):
        mask = y == c
        plt.scatter(X2[mask, 0], X2[mask, 1], c=colors[c], label=labels_map[c], alpha=0.7)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_3d(X3, title, filename):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    for c in np.unique(y):
        mask = y == c
        ax.scatter(X3[mask, 0], X3[mask, 1], X3[mask, 2], c=colors[c], label=labels_map[c], alpha=0.7)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)
plot_2d(X_pca2, 'PCA (2 components)', 'pca_2d.png')

pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)
plot_3d(X_pca3, 'PCA (3 components)', 'pca_3d.png')

print("PCA explained variance ratio (2 comp):", pca2.explained_variance_ratio_.sum())
print("PCA explained variance ratio (3 comp):", pca3.explained_variance_ratio_.sum())

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

def train_autoencoder(X_scaled, latent_dim, epochs=300, lr=0.001):
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    model = Autoencoder(X_scaled.shape[1], latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output, _ = model(X_tensor)
        loss = criterion(output, X_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Latent dim {latent_dim}, epoch {epoch+1}/{epochs}, loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        _, latent = model(X_tensor)
    return latent.numpy(), loss.item()

X_ae2, loss2 = train_autoencoder(X_scaled, 2)
plot_2d(X_ae2, 'Autoencoder (2 neurons)', 'autoencoder_2d.png')
print("Autoencoder final reconstruction loss (2 neurons):", loss2)

X_ae3, loss3 = train_autoencoder(X_scaled, 3)
plot_3d(X_ae3, 'Autoencoder (3 neurons)', 'autoencoder_3d.png')
print("Autoencoder final reconstruction loss (3 neurons):", loss3)

perplexities = [20, 30, 40, 50, 60]

for p in perplexities:
    tsne2 = TSNE(n_components=2, perplexity=p, init='pca', random_state=42)
    X_tsne2 = tsne2.fit_transform(X_scaled)
    plot_2d(X_tsne2, f't-SNE 2D (perplexity={p})', f'tsne_2d_perp{p}.png')

for p in perplexities:
    tsne3 = TSNE(n_components=3, perplexity=p, init='pca', random_state=42)
    X_tsne3 = tsne3.fit_transform(X_scaled)
    plot_3d(X_tsne3, f't-SNE 3D (perplexity={p})', f'tsne_3d_perp{p}.png')

