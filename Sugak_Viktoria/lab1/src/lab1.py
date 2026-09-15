import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

columns = [
    "ID", "Category", "Age", "Sex", "ALB", "ALP",
    "ALT", "AST", "BIL", "CHE", "CHOL", "CREA",
    "GGT", "PROT"
]

data = pd.read_csv("1.csv", header=None, names=["raw"])
data = data["raw"].str.split(",", expand=True)
data.columns = columns
data = data.iloc[1:].copy()

features = [
    "Age", "ALB", "ALP", "ALT", "AST",
    "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"
]

X = data[features].apply(pd.to_numeric, errors="coerce")
y = data["Category"].str.replace('"', '')

valid = X.notna().all(axis=1)
X = X[valid]
y = y[valid]

X_std = (X - X.mean()) / X.std()

cov_matrix = np.cov(X_std, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

X_pca_manual = X_std.values @ eigenvectors[:, :3]

explained_variance = eigenvalues / eigenvalues.sum()

print("Размер исходных данных:", X.shape)

print("\nСобственные значения:")
print(eigenvalues)

print("\nДоля объяснённой дисперсии:")
print(explained_variance)

print("\nРазмерность после PCA:", X_pca_manual.shape)

pca = PCA(n_components=3)
X_pca_sklearn = pca.fit_transform(X_std)

print("\nPCA sklearn:")
print("Собственные значения:")
print(pca.explained_variance_)

print("\nДоля объяснённой дисперсии:")
print(pca.explained_variance_ratio_)

loss_2 = 1 - pca.explained_variance_ratio_[:2].sum()
loss_3 = 1 - pca.explained_variance_ratio_[:3].sum()

print("\nПотери при использовании 2 компонент:", loss_2)
print("Потери при использовании 3 компонент:", loss_3)

classes = y.unique()

plt.figure(figsize=(9, 6))

for cls in classes:
    mask = y == cls
    plt.scatter(
        X_pca_sklearn[mask, 0],
        X_pca_sklearn[mask, 1],
        label=cls,
        alpha=0.7
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Первые две главные компоненты")
plt.legend()
plt.grid()
plt.show()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

for cls in classes:
    mask = y == cls
    ax.scatter(
        X_pca_sklearn[mask, 0],
        X_pca_sklearn[mask, 1],
        X_pca_sklearn[mask, 2],
        label=cls,
        alpha=0.7
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Первые три главные компоненты")
ax.legend()

plt.show()