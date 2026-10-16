import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler

print('Загрузка датасета Rice (Cammeo and Osmancik) ')
url = 'https://archive.ics.uci.edu/static/public/545/rice+cammeo+and+osmancik.zip'


try:
    from ucimlrepo import fetch_ucirepo

    rice = fetch_ucirepo(id=545)
    X_raw = rice.data.features
    y_raw = rice.data.targets['Class']
except ImportWarning:

    df = pd.read_csv(
        'https://raw.githubusercontent.com/jbrownlee/Datasets/master/rice_cammeo_osmancik.csv'
    )
    X_raw = df.iloc[:, :-1]
    y_raw = df.iloc[:, -1]


le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

n_features = X_scaled.shape[1]
print(
    f'Загружено {X_scaled.shape[0]} образцов, {n_features} признаков, {len(np.unique(y))} класса.'
)


class Autoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),  # Узкое горлышко (bottleneck)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def train_autoencoder(X_data, latent_dim, epochs=100, lr=0.01):
    tensor_x = torch.FloatTensor(X_data)
    model = Autoencoder(input_dim=X_data.shape[1], latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed, _ = model(tensor_x)
        loss = criterion(reconstructed, tensor_x)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, latent_features = model(tensor_x)
    return latent_features.numpy()


print('Обучение Автоэнкодеров (2D и 3D) ')
X_ae_2d = train_autoencoder(X_scaled, latent_dim=2)
X_ae_3d = train_autoencoder(X_scaled, latent_dim=3)


print(' Вычисление t-SNE (2D и 3D) ')
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne_2d = tsne_2d.fit_transform(X_scaled)

tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
X_tsne_3d = tsne_3d.fit_transform(X_scaled)

print(' Вычисление PCA ')
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))


scatter1 = axes[0].scatter(
    X_ae_2d[:, 0],
    X_ae_2d[:, 1],
    c=y,
    cmap='viridis',
    s=15,
    alpha=0.7,
    edgecolor='none',
)
axes[0].set_title('Автоэнкодер (2D)')
axes[0].set_xlabel('Latent 1')
axes[0].set_ylabel('Latent 2')
axes[0].grid(True)


scatter2 = axes[1].scatter(
    X_tsne_2d[:, 0],
    X_tsne_2d[:, 1],
    c=y,
    cmap='viridis',
    s=15,
    alpha=0.7,
    edgecolor='none',
)
axes[1].set_title('t-SNE (2D)')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].grid(True)

scatter3 = axes[2].scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=y,
    cmap='viridis',
    s=15,
    alpha=0.7,
    edgecolor='none',
)
axes[2].set_title('PCA (2D)')
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')
axes[2].grid(True)


cbar = fig.colorbar(scatter3, ax=axes, orientation='horizontal', pad=0.15, shrink=0.6)
cbar.set_ticks([0.25, 0.75])
cbar.set_ticklabels(class_names)
plt.suptitle(
    'Сравнение методов понижения размерности 2D (Вариант 3: Rice Dataset)',
    fontsize=14,
)
plt.show()

fig = plt.figure(figsize=(18, 6))


ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(
    X_ae_3d[:, 0],
    X_ae_3d[:, 1],
    X_ae_3d[:, 2],
    c=y,
    cmap='viridis',
    s=15,
    alpha=0.7,
)
ax1.set_title('Автоэнкодер (3D)')
ax1.set_xlabel('Latent 1')
ax1.set_ylabel('Latent 2')
ax1.set_zlabel('Latent 3')

# t-SNE 3D
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(
    X_tsne_3d[:, 0],
    X_tsne_3d[:, 1],
    X_tsne_3d[:, 2],
    c=y,
    cmap='viridis',
    s=15,
    alpha=0.7,
)
ax2.set_title('t-SNE (3D)')
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
ax2.set_zlabel('t-SNE 3')

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(
    X_pca_3d[:, 0],
    X_pca_3d[:, 1],
    X_pca_3d[:, 2],
    c=y,
    cmap='viridis',
    s=15,
    alpha=0.7,
)
ax3.set_title('PCA (3D)')
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_zlabel('PC3')

plt.suptitle(
    'Сравнение методов понижения размерности 3D (Вариант 3: Rice Dataset)',
    fontsize=14,
)
plt.tight_layout()
plt.show()