import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('seeds_dataset.txt', sep=r'\s+', header=None, na_values=[''])


df.fillna(df.mean(), inplace=True)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Ручная реализация PCA (NumPy)
cov_matrix = np.cov(X_scaled, rowvar=False)

eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

eigen_values = eigen_values.real
eigen_vectors = eigen_vectors.real

sort_index = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[sort_index]
eigen_vectors = eigen_vectors[:, sort_index]

PC2_manual = np.dot(X_scaled, eigen_vectors[:, 0:2])
PC3_manual = np.dot(X_scaled, eigen_vectors[:, 0:3])

# Использование scikit-learn
pca2 = PCA(n_components=2)
PC2_sklearn = pca2.fit_transform(X_scaled)

pca3 = PCA(n_components=3)
PC3_sklearn = pca3.fit_transform(X_scaled)

full_info = eigen_values.sum()

reduce_info_2 = eigen_values[0:2].sum()
loss_2 = 100 - (reduce_info_2 / full_info * 100)

reduce_info_3 = eigen_values[0:3].sum()
loss_3 = 100 - (reduce_info_3 / full_info * 100)

print("--- Расчет потерь информативности ---")
print(f"Потери информативности при переходе к 2 компонентам: {loss_2:.2f}%")
print(f"Потери информативности при переходе к 3 компонентам: {loss_3:.2f}%\n")


classes = np.unique(y)
colors = ['red', 'green', 'blue']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, cls in enumerate(classes):
    mask = (y == cls)
    axes[0].scatter(PC2_manual[mask, 0], PC2_manual[mask, 1],
                    c=colors[i], label=f'Class {int(cls)}', alpha=0.7)
axes[0].set_title('Метод 1: Ручной PCA (NumPy) - 2 компоненты')
axes[0].set_xlabel('Главная компонента 1 (PC1)')
axes[0].set_ylabel('Главная компонента 2 (PC2)')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

for i, cls in enumerate(classes):
    mask = (y == cls)
    axes[1].scatter(PC2_sklearn[mask, 0], PC2_sklearn[mask, 1],
                    c=colors[i], label=f'Class {int(cls)}', alpha=0.7)
axes[1].set_title('Метод 2: Scikit-Learn PCA - 2 компоненты')
axes[1].set_xlabel('Главная компонента 1 (PC1)')
axes[1].set_ylabel('Главная компонента 2 (PC2)')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(14, 6))


ax1 = fig.add_subplot(121, projection='3d')
for i, cls in enumerate(classes):
    mask = (y == cls)
    ax1.scatter(PC3_manual[mask, 0], PC3_manual[mask, 1], PC3_manual[mask, 2],
                c=colors[i], label=f'Class {int(cls)}', alpha=0.7)
ax1.set_title('Метод 1: Ручной PCA (NumPy) - 3 компоненты')
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2'); ax1.set_zlabel('PC3')
ax1.legend()


ax2 = fig.add_subplot(122, projection='3d')
for i, cls in enumerate(classes):
    mask = (y == cls)
    ax2.scatter(PC3_sklearn[mask, 0], PC3_sklearn[mask, 1], PC3_sklearn[mask, 2],
                c=colors[i], label=f'Class {int(cls)}', alpha=0.7)
ax2.set_title('Метод 2: Scikit-Learn PCA - 3 компоненты')
ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2'); ax2.set_zlabel('PC3')
ax2.legend()

plt.tight_layout()
plt.show()