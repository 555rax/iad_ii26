import zipfile
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with zipfile.ZipFile('seeds.zip', 'r') as z:
    z.extractall('seeds_data')

columns = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'groove_length', 'class']
df = pd.read_csv('seeds_data/seeds_dataset.txt', sep=r'\s+', names=columns, header=None)

print("Пропущенные значения по столбцам до обработки:")
print(df.isna().sum())

df = df.fillna(df.mean(numeric_only=True))

X = df.drop(columns=['class']).values
y = df['class'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cov_matrix = np.cov(X_scaled.T)
eig_values, eig_vectors = np.linalg.eigh(cov_matrix)

idx = np.argsort(eig_values)[::-1]
eig_values = eig_values[idx]
eig_vectors = eig_vectors[:, idx]

W2 = eig_vectors[:, :2]
W3 = eig_vectors[:, :3]

X_pca2_manual = X_scaled.dot(W2)
X_pca3_manual = X_scaled.dot(W3)

colors = {1: 'r', 2: 'g', 3: 'b'}
labels_map = {1: 'Class 1', 2: 'Class 2', 3: 'Class 3'}

fig = plt.figure(figsize=(16, 6))

ax1 = fig.add_subplot(1, 2, 1)
for c in np.unique(y):
    mask = y == c
    ax1.scatter(X_pca2_manual[mask, 0], X_pca2_manual[mask, 1], c=colors[c], label=labels_map[c])
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title('Manual PCA (2D)')
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
for c in np.unique(y):
    mask = y == c
    ax2.scatter(X_pca3_manual[mask, 0], X_pca3_manual[mask, 1], X_pca3_manual[mask, 2], c=colors[c], label=labels_map[c])
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_zlabel('PC3')
ax2.set_title('Manual PCA (3D)')
ax2.legend()

plt.tight_layout()
plt.savefig('manual_pca.png')
plt.show()

pca2 = PCA(n_components=2)
X_pca2_sklearn = pca2.fit_transform(X_scaled)

pca3 = PCA(n_components=3)
X_pca3_sklearn = pca3.fit_transform(X_scaled)

fig = plt.figure(figsize=(16, 6))

ax1 = fig.add_subplot(1, 2, 1)
for c in np.unique(y):
    mask = y == c
    ax1.scatter(X_pca2_sklearn[mask, 0], X_pca2_sklearn[mask, 1], c=colors[c], label=labels_map[c])
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title('sklearn PCA (2D)')
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
for c in np.unique(y):
    mask = y == c
    ax2.scatter(X_pca3_sklearn[mask, 0], X_pca3_sklearn[mask, 1], X_pca3_sklearn[mask, 2], c=colors[c], label=labels_map[c])
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_zlabel('PC3')
ax2.set_title('sklearn PCA (3D)')
ax2.legend()

plt.tight_layout()
plt.savefig('sklearn_pca.png')
plt.show()

total_variance = np.sum(eig_values)
explained_2 = np.sum(eig_values[:2]) / total_variance
explained_3 = np.sum(eig_values[:3]) / total_variance
loss_2 = 1 - explained_2
loss_3 = 1 - explained_3

print("\nСобственные значения (ручной метод):", eig_values)
print("Собственные значения sklearn (explained_variance_, 3 комп):", pca3.explained_variance_)

print("\nДоля объясненной дисперсии (2 компоненты, ручной метод):", explained_2)
print("Доля объясненной дисперсии (2 компоненты, sklearn):", pca2.explained_variance_ratio_.sum())

print("\nДоля объясненной дисперсии (3 компоненты, ручной метод):", explained_3)
print("Доля объясненной дисперсии (3 компоненты, sklearn):", pca3.explained_variance_ratio_.sum())

print("\nПотери информации при проекции на 2 компоненты:", loss_2)
print("Потери информации при проекции на 3 компоненты:", loss_3)

