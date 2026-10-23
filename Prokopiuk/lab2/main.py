import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Load data
column_names = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
    'ring-type', 'spore-print-color', 'population', 'habitat'
]

data = pd.read_csv('agaricus-lepiota.data', header=None, names=column_names)

X = data.drop('class', axis=1)
y = data['class']
y_encoded = (y == 'e').astype(int).values

# One-hot encoding for categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)
print(f"Data shape after OHE: {X_encoded.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=RANDOM_SEED, stratify=y_encoded
)


# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.encoder(x), self.decoder(self.encoder(x))


def train_ae(latent_dim):
    model = Autoencoder(input_dim=X_train.shape[1], latent_dim=latent_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=64, shuffle=True
    )

    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in train_loader:
            data_batch = batch[0].to(DEVICE)
            optimizer.zero_grad()
            _, decoded = model(data_batch)
            loss = criterion(decoded, data_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50, Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        X_all = torch.tensor(X_encoded, dtype=torch.float32).to(DEVICE)
        latent, _ = model(X_all)
    return latent.cpu().numpy()


# Autoencoder 2D
print("\nTraining Autoencoder 2D...")
ae_2d = train_ae(2)

plt.figure(figsize=(10, 8))
plt.scatter(ae_2d[:, 0], ae_2d[:, 1], c=y_encoded, cmap='coolwarm', alpha=0.6, s=15)
plt.title('Autoencoder 2D')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Poisonous (p)', markerfacecolor='tab:blue', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Edible (e)', markerfacecolor='tab:red', markersize=8)
])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Autoencoder 3D
print("\nTraining Autoencoder 3D...")
ae_3d = train_ae(3)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ae_3d[:, 0], ae_3d[:, 1], ae_3d[:, 2], c=y_encoded, cmap='coolwarm', alpha=0.6, s=15)
ax.set_title('Autoencoder 3D')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
plt.tight_layout()
plt.show()


# t-SNE with different perplexity values
print("\nRunning t-SNE...")
X_subset = X_encoded[::2]
y_subset = y_encoded[::2]

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
perplexities = [20, 40, 50, 60]

for ax, perp in zip(axes.flatten(), perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, init='pca', random_state=RANDOM_SEED, n_jobs=-1)
    result = tsne.fit_transform(X_subset)
    ax.scatter(result[:, 0], result[:, 1], c=y_subset, cmap='coolwarm', alpha=0.6, s=15)
    ax.set_title(f't-SNE (perplexity={perp})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# t-SNE 3D
print("\nRunning t-SNE 3D...")
tsne_3d = TSNE(n_components=3, perplexity=30, init='pca', random_state=RANDOM_SEED, n_jobs=-1)
tsne_3d_result = tsne_3d.fit_transform(X_subset)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne_3d_result[:, 0], tsne_3d_result[:, 1], tsne_3d_result[:, 2],
           c=y_subset, cmap='coolwarm', alpha=0.6, s=15)
ax.set_title('t-SNE 3D')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
plt.tight_layout()
plt.show()


# PCA
print("\nRunning PCA...")
pca_2d = PCA(n_components=2, random_state=RANDOM_SEED)
pca_2d_result = pca_2d.fit_transform(X_encoded)

plt.figure(figsize=(10, 8))
plt.scatter(pca_2d_result[:, 0], pca_2d_result[:, 1], c=y_encoded, cmap='coolwarm', alpha=0.6, s=15)
plt.title(f'PCA 2D (variance: {pca_2d.explained_variance_ratio_.sum()*100:.1f}%)')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Poisonous (p)', markerfacecolor='tab:blue', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', label='Edible (e)', markerfacecolor='tab:red', markersize=8)
])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# PCA 3D
pca_3d = PCA(n_components=3, random_state=RANDOM_SEED)
pca_3d_result = pca_3d.fit_transform(X_encoded)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_3d_result[:, 0], pca_3d_result[:, 1], pca_3d_result[:, 2],
           c=y_encoded, cmap='coolwarm', alpha=0.6, s=15)
ax.set_title('PCA 3D')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.tight_layout()
plt.show()

print("\nDone.")