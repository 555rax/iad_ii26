import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# === 1. Загрузка данных ===
# Укажи путь к CSV внутри архива wholesale+customers.zip
# Например: data = pd.read_csv("Wholesale customers data.csv")
data = pd.read_csv("Wholesale customers data.csv")

# Класс для визуализации — колонка "Region"
labels = data["Region"]
X = data.drop(columns=["Region"])

# === 2. PCA вручную через numpy.linalg.eig ===
# Центрируем данные
X_centered = X - X.mean(axis=0)

# Ковариационная матрица
cov_matrix = np.cov(X_centered, rowvar=False)

# Собственные значения и векторы
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# Сортировка по убыванию
idx = eig_vals.argsort()[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

# Проекция на первые 2 и 3 компоненты
X_pca_manual_2D = np.dot(X_centered, eig_vecs[:, :2])
X_pca_manual_3D = np.dot(X_centered, eig_vecs[:, :3])

# === 3. PCA через sklearn ===
pca = PCA(n_components=3)
X_pca_sklearn = pca.fit_transform(X)

# === 4. Визуализация ===
def plot_2D(data_2d, labels, title):
    plt.figure(figsize=(8,6))
    for region in np.unique(labels):
        plt.scatter(data_2d[labels==region, 0],
                    data_2d[labels==region, 1],
                    label=f"Region {region}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_3D(data_3d, labels, title):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    for region in np.unique(labels):
        ax.scatter(data_3d[labels==region, 0],
                   data_3d[labels==region, 1],
                   data_3d[labels==region, 2],
                   label=f"Region {region}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend()
    plt.show()

# Визуализация вручную
plot_2D(X_pca_manual_2D, labels.values, "PCA вручную (2D)")
plot_3D(X_pca_manual_3D, labels.values, "PCA вручную (3D)")

# Визуализация sklearn
plot_2D(X_pca_sklearn[:, :2], labels.values, "PCA sklearn (2D)")
plot_3D(X_pca_sklearn, labels.values, "PCA sklearn (3D)")

# === 5. Потери информации ===
explained_variance_ratio = pca.explained_variance_ratio_
print("Доля объяснённой дисперсии по компонентам:", explained_variance_ratio)
print("Суммарная сохранённая дисперсия:", explained_variance_ratio.sum())
print("Потери информации:", 1 - explained_variance_ratio.sum())
