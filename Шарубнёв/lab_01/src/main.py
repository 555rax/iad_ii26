import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv("E:\PythonProjects\IAD1\heart_failure_clinical_records_dataset.csv")

print("Первые строки:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print("\nПропуски:")
print(df.isna().sum())


y = df["DEATH_EVENT"]
X = df.drop(columns=["DEATH_EVENT"])

X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cov_matrix = np.cov(X_scaled, rowvar=False)

eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

idx = np.argsort(eig_vals)[::-1]
eig_vals_sorted = eig_vals[idx]
eig_vecs_sorted = eig_vecs[:, idx]

W2 = eig_vecs_sorted[:, :2]
Z2_manual = X_scaled @ W2

W3 = eig_vecs_sorted[:, :3]
Z3_manual = X_scaled @ W3

pca2 = PCA(n_components=2)
Z2_sklearn = pca2.fit_transform(X_scaled)

pca3 = PCA(n_components=3)
Z3_sklearn = pca3.fit_transform(X_scaled)

pca_full = PCA()
pca_full.fit(X_scaled)

eig_vals_sklearn = pca_full.explained_variance_
explained_ratio_sklearn = pca_full.explained_variance_ratio_

print("\nСобственные значения sklearn:")
print(eig_vals_sklearn)

print("\nДоля объяснённой дисперсии sklearn:")
print(explained_ratio_sklearn)

total_var = np.sum(eig_vals_sorted)

explained_var_2 = np.sum(eig_vals_sorted[:2]) / total_var
explained_var_3 = np.sum(eig_vals_sorted[:3]) / total_var

loss_2 = 1 - explained_var_2
loss_3 = 1 - explained_var_3

print(f"\nДоля сохранённой дисперсии (2 компоненты): {explained_var_2:.4f}")
print(f"Потери (2 компоненты): {loss_2:.4f}")

print(f"\nДоля сохранённой дисперсии (3 компоненты): {explained_var_3:.4f}")
print(f"Потери (3 компоненты): {loss_3:.4f}")

plt.figure(figsize=(8, 6))

for cls, color, label in [(0, "blue", "No death"), (1, "red", "Death")]:
    mask = (y == cls)
    plt.scatter(Z2_sklearn[mask, 0], Z2_sklearn[mask, 1],
                c=color, label=label, alpha=0.7)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA (2 компоненты)")
plt.legend()
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for cls, color, label in [(0, "blue", "No death"), (1, "red", "Death")]:
    mask = (y == cls)
    ax.scatter(Z3_sklearn[mask, 0], Z3_sklearn[mask, 1], Z3_sklearn[mask, 2],
               c=color, label=label, alpha=0.7)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA (3 компоненты)")
ax.legend()
plt.show()
