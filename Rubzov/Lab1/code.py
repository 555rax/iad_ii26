import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('Exasens.csv')

data = df.iloc[2:].copy()

data.columns = [
    'Diagnosis', 'ID', 'Imaginary_Min', 'Imaginary_Avg',
    'Real_Min', 'Real_Avg', 'Gender', 'Age', 'Smoking',
    'Col9', 'Col10', 'Col11', 'Col12'
]

y = data['Diagnosis'].values

X_raw = data.drop(columns=['Diagnosis', 'ID', 'Col9', 'Col10', 'Col11', 'Col12'])

X_numeric = X_raw.apply(pd.to_numeric, errors='coerce')

X_imputed = X_numeric.fillna(X_numeric.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

cov_matrix = np.cov(X_scaled, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sort_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_idx]
eigenvectors = eigenvectors[:, sort_idx]

X_manual_2d = X_scaled @ eigenvectors[:, :2]
X_manual_3d = X_scaled @ eigenvectors[:, :3]

pca_2d = PCA(n_components=2)
X_sklearn_2d = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3)
X_sklearn_3d = pca_3d.fit_transform(X_scaled)

total_var = np.sum(eigenvalues)

var_2d = np.sum(eigenvalues[:2]) / total_var
loss_2d = (1 - var_2d) * 100

var_3d = np.sum(eigenvalues[:3]) / total_var
loss_3d = (1 - var_3d) * 100

print("="*60)
print("РЕЗУЛЬТАТЫ РАСЧЕТА ПОТЕРЬ ИНФОРМАЦИИ (PCA):")
print(f"Собственные значения: {np.round(eigenvalues, 4)}")
print(f"Объясненная дисперсия (2D): {var_2d*100:.2f}% | Потери: {loss_2d:.2f}%")
print(f"Объясненная дисперсия (3D): {var_3d*100:.2f}% | Потери: {loss_3d:.2f}%")
print("="*60)

unique_classes = np.unique(y)
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for cls, color in zip(unique_classes, colors):
    mask = (y == cls)
    axes[0].scatter(X_manual_2d[mask, 0], X_manual_2d[mask, 1], label=cls, color=color, alpha=0.7)
    axes[1].scatter(X_sklearn_2d[mask, 0], X_sklearn_2d[mask, 1], label=cls, color=color, alpha=0.7)

axes[0].set_title("2D PCA (Ручной способ: numpy.linalg.eig)")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend()
axes[0].grid(True)

axes[1].set_title("2D PCA (sklearn.decomposition.PCA)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

for cls, color in zip(unique_classes, colors):
    mask = (y == cls)
    ax1.scatter(X_manual_3d[mask, 0], X_manual_3d[mask, 1], X_manual_3d[mask, 2], label=cls, color=color, alpha=0.7)
    ax2.scatter(X_sklearn_3d[mask, 0], X_sklearn_3d[mask, 1], X_sklearn_3d[mask, 2], label=cls, color=color, alpha=0.7)

ax1.set_title("3D PCA (Ручной способ)")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_zlabel("PC3")
ax1.legend()

ax2.set_title("3D PCA (sklearn)")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_zlabel("PC3")
ax2.legend()

plt.tight_layout()
plt.show()
