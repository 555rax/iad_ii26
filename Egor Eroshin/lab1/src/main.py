import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('seeds_dataset.txt', header=None, sep=r'\s+')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cov_matrix = np.cov(X_scaled, rowvar=False)
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

eigen_values = eigen_values.real
eigen_vectors = eigen_vectors.real
sorted_idx = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[sorted_idx]
eigen_vectors = eigen_vectors[:, sorted_idx]

X_pca_manual_2d = np.dot(X_scaled, eigen_vectors[:, :2])
X_pca_manual_3d = np.dot(X_scaled, eigen_vectors[:, :3])

total_var = np.sum(eigen_values)
var_2g = np.sum(eigen_values[:2]) / total_var
var_3g = np.sum(eigen_values[:3]) / total_var

print('=== Ручной PCA ===')
print(
    f'Доля объясненной дисперсии (2 ГК): {var_2g:.4f} ({var_2g*100:.2f}%)'
)
print(f'Потери (2 ГК): {1 - var_2g:.4f} ({(1 - var_2g)*100:.2f}%)')
print(
    f'Доля объясненной дисперсии (3 ГК): {var_3g:.4f} ({var_3g*100:.2f}%)'
)
print(f'Потери (3 ГК): {1 - var_3g:.4f} ({(1 - var_3g)*100:.2f}%)')

pca_sk_2d = PCA(n_components=2)
X_pca_sk_2d = pca_sk_2d.fit_transform(X_scaled)

pca_sk_3d = PCA(n_components=3)
X_pca_sk_3d = pca_sk_3d.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

scatter1 = axes[0].scatter(
    X_pca_manual_2d[:, 0],
    X_pca_manual_2d[:, 1],
    c=y,
    cmap='viridis',
    s=30,
    edgecolor='k',
)
axes[0].set_title('Ручной PCA — 2 компоненты')
axes[0].set_xlabel('PC1 (manual)')
axes[0].set_ylabel('PC2 (manual)')
axes[0].grid(True)

scatter2 = axes[1].scatter(
    X_pca_sk_2d[:, 0],
    X_pca_sk_2d[:, 1],
    c=y,
    cmap='viridis',
    s=30,
    edgecolor='k',
)
axes[1].set_title('sklearn PCA — 2 компоненты')
axes[1].set_xlabel('PC1 (sklearn)')
axes[1].set_ylabel('PC2 (sklearn)')
axes[1].grid(True)

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(
    X_pca_manual_3d[:, 0],
    X_pca_manual_3d[:, 1],
    X_pca_manual_3d[:, 2],
    c=y,
    cmap='viridis',
    s=30,
)
ax1.set_title('Ручной PCA — 3 компоненты')
ax1.set_xlabel('PC1 (manual)')
ax1.set_ylabel('PC2 (manual)')
ax1.set_zlabel('PC3 (manual)')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(
    X_pca_sk_3d[:, 0],
    X_pca_sk_3d[:, 1],
    X_pca_sk_3d[:, 2],
    c=y,
    cmap='viridis',
    s=30,
)
ax2.set_title('sklearn PCA — 3 компоненты')
ax2.set_xlabel('PC1 (sklearn)')
ax2.set_ylabel('PC2 (sklearn)')
ax2.set_zlabel('PC3 (sklearn)')

plt.tight_layout()
plt.show()