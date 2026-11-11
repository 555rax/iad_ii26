import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

data = load_breast_cancer()
X = data.data
y = data.target
target_names = data.target_names

X_scaled = StandardScaler().fit_transform(X)

def build_autoencoder(X, encoding_dim):
    input_layer = Input(shape=(X.shape[1],))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(X.shape[1], activation='linear')(encoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X, X, epochs=100, batch_size=32, shuffle=True, verbose=0)
    return encoder.predict(X), history.history['loss']

ae_2d, loss_2d = build_autoencoder(X_scaled, 2)
ae_3d, _ = build_autoencoder(X_scaled, 3)

fig = plt.figure(figsize=(18, 5))
fig.suptitle('Autoencoder Visualization', fontsize=16)

ax1 = fig.add_subplot(131)
scatter1 = ax1.scatter(ae_2d[:, 0], ae_2d[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.8)
ax1.set_title('Autoencoder 2D')
ax1.set_xlabel('Latent Dim 1')
ax1.set_ylabel('Latent Dim 2')
ax1.grid(True, alpha=0.3)
handles, labels = scatter1.legend_elements()
ax1.legend(handles, target_names, title="Classes")

ax2 = fig.add_subplot(132)
ax2.plot(loss_2d, color='purple', linewidth=2)
ax2.set_title('Autoencoder Training Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE Loss')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(133, projection='3d')
scatter3 = ax3.scatter(ae_3d[:, 0], ae_3d[:, 1], ae_3d[:, 2], c=y, cmap='coolwarm', edgecolors='k', alpha=0.8)
ax3.set_title('Autoencoder 3D')
ax3.set_xlabel('Latent 1')
ax3.set_ylabel('Latent 2')
ax3.set_zlabel('Latent 3')

plt.tight_layout()
plt.show()

pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_scaled)
pca_2d = X_pca[:, :2]
pca_3d = X_pca[:, :3]

fig = plt.figure(figsize=(18, 5))
fig.suptitle('PCA Visualization', fontsize=16)

ax1 = fig.add_subplot(131)
ax1.bar(range(1, 11), pca.explained_variance_ratio_[:10], alpha=0.7, color='teal', label='Individual')
ax1.plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_[:10]), marker='o', color='red', label='Cumulative')
ax1.set_title('PCA Explained Variance')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_xticks(range(1, 11))
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(132)
scatter2 = ax2.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.8)
ax2.set_title('PCA 2D')
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.grid(True, alpha=0.3)
handles, labels = scatter2.legend_elements()
ax2.legend(handles, target_names, title="Classes")

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2], c=y, cmap='coolwarm', edgecolors='k', alpha=0.8)
ax3.set_title('PCA 3D')
ax3.set_xlabel('PC 1')
ax3.set_ylabel('PC 2')
ax3.set_zlabel('PC 3')

plt.tight_layout()
plt.show()

tsne_2d = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
tsne_3d = TSNE(n_components=3, random_state=42).fit_transform(X_scaled)

fig = plt.figure(figsize=(12, 5))
fig.suptitle('t-SNE Visualization', fontsize=16)

ax1 = fig.add_subplot(121)
scatter1 = ax1.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.8)
ax1.set_title('t-SNE 2D')
ax1.set_xlabel('t-SNE Dim 1')
ax1.set_ylabel('t-SNE Dim 2')
ax1.grid(True, alpha=0.3)
handles, labels = scatter1.legend_elements()
ax1.legend(handles, target_names, title="Classes")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(tsne_3d[:, 0], tsne_3d[:, 1], tsne_3d[:, 2], c=y, cmap='coolwarm', edgecolors='k', alpha=0.8)
ax2.set_title('t-SNE 3D')
ax2.set_xlabel('Dim 1')
ax2.set_ylabel('Dim 2')
ax2.set_zlabel('Dim 3')

plt.tight_layout()
plt.show()