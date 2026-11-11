import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

X, y = load_breast_cancer(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

def build_autoencoder(X, encoding_dim):
    input_layer = Input(shape=(X.shape[1],))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(X.shape[1], activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=100, batch_size=32, shuffle=True, verbose=0)
    return encoder.predict(X)

ae_2d = build_autoencoder(X_scaled, 2)
ae_3d = build_autoencoder(X_scaled, 3)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(ae_2d[:, 0], ae_2d[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.title('Autoencoder 2D')
plt.subplot(1, 2, 2)
ax = plt.axes(projection='3d')
ax.scatter(ae_3d[:, 0], ae_3d[:, 1], ae_3d[:, 2], c=y, cmap='viridis')
ax.set_title('Autoencoder 3D')
plt.show()

pca_2d = PCA(n_components=2).fit_transform(X_scaled)
pca_3d = PCA(n_components=3).fit_transform(X_scaled)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.title('PCA 2D')
plt.subplot(1, 2, 2)
ax = plt.axes(projection='3d')
ax.scatter(pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2], c=y, cmap='viridis')
ax.set_title('PCA 3D')
plt.show()

tsne_2d = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
tsne_3d = TSNE(n_components=3, random_state=42).fit_transform(X_scaled)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.title('t-SNE 2D')
plt.subplot(1, 2, 2)
ax = plt.axes(projection='3d')
ax.scatter(tsne_3d[:, 0], tsne_3d[:, 1], tsne_3d[:, 2], c=y, cmap='viridis')
ax.set_title('t-SNE 3D')
plt.show()
