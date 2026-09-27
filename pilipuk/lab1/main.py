import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('seeds_dataset.txt', sep=r'\s+', header=None)

if df.isnull().sum().sum() > 0:
    df = df.fillna(df.median())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

cov_matrix = np.cov(X.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_manual = eigenvalues[sorted_indices]
eigenvectors_manual = eigenvectors[:, sorted_indices]

W_2d = eigenvectors_manual[:, :2]
X_pca_manual_2d = X.dot(W_2d)

W_3d = eigenvectors_manual[:, :3]
X_pca_manual_3d = X.dot(W_3d)

pca_2d = PCA(n_components=2)
X_pca_sklearn_2d = pca_2d.fit_transform(X)

pca_3d = PCA(n_components=3)
X_pca_sklearn_3d = pca_3d.fit_transform(X)

colors = {1: 'red', 2: 'green', 3: 'blue'}
labels = {1: 'Vermillion (Class 1)', 2: 'Fade To Black (Class 2)', 3: 'QWERTY (Class 3)'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for cls in np.unique(y):
    axes[0].scatter(X_pca_manual_2d[y == cls, 0], X_pca_manual_2d[y == cls, 1],
                    c=colors[cls], label=labels[cls], alpha=0.7)
axes[0].set_title('2D PCA (HandMake)')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend()
axes[0].grid(True)

for cls in np.unique(y):
    axes[1].scatter(X_pca_sklearn_2d[y == cls, 0], X_pca_sklearn_2d[y == cls, 1],
                    c=colors[cls], label=labels[cls], alpha=0.7)
axes[1].set_title('2D PCA (sklearn)')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('2 PCA', fontsize=14)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
for cls in np.unique(y):
    ax1.scatter(X_pca_manual_3d[y == cls, 0], 
                X_pca_manual_3d[y == cls, 1], 
                X_pca_manual_3d[y == cls, 2],
                c=colors[cls], label=labels[cls], alpha=0.7)
ax1.set_title('3D PCA (HandMake)')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')
ax1.legend()

ax2 = fig.add_subplot(122, projection='3d')
for cls in np.unique(y):
    ax2.scatter(X_pca_sklearn_3d[y == cls, 0], 
                X_pca_sklearn_3d[y == cls, 1], 
                X_pca_sklearn_3d[y == cls, 2],
                c=colors[cls], label=labels[cls], alpha=0.7)
ax2.set_title('3D PCA (sklearn)')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_zlabel('PC3')
ax2.legend()

plt.suptitle('3 PCA', fontsize=14)
plt.tight_layout()
plt.show()

total_variance = np.sum(eigenvalues_manual)

expvar2d = np.sum(eigenvalues_manual[:2]) / total_variance
loss_2d = (1 - expvar2d) * 100

expvar3d = np.sum(eigenvalues_manual[:3]) / total_variance
loss_3d = (1 - expvar3d) * 100

print(f"Explained Variance (2 comp): {expvar2d * 100:.2f}%")
print(f"Information Loss (2 comp): {loss_2d:.2f}%")
print(f"Explained Variance (3 comp): {expvar3d * 100:.2f}%")
print(f"Information Loss (3 comp): {loss_3d:.2f}%")