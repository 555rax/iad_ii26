import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv("lab1/hcvdat0.csv")

y = data["Category"]

X = data.drop(columns=["Unnamed: 0", 'Category'])
X["Sex"] = X["Sex"].map({"m" : 0, "f" : 1})
X = X.fillna(X.mean())
X = (X - X.mean()) / X.std()

cov = np.cov(X, rowvar=False)
eigenval, eigwnvec = np.linalg.eig(cov)

idx = np.argsort(eigenval)[::-1]
eigenval = eigenval[idx]
eigwnvec = eigwnvec[:, idx]

PC = eigwnvec[:, :3] # PC1, PC2, PC3

X_pca = X.values @ PC
X_pca = X_pca.real

explained = eigenval.real / eigenval.real.sum()

pca = PCA(n_components=3)
X_pca_sklearn = pca.fit_transform(X)

# for i in range(3):
#     if np.dot(PC[:, i], pca.components_[i]) < 0:
#         X_pca[:, i] *= -1

loss_2 = 1 - explained[:2].sum()
loss_3 = 1 - explained[:3].sum()

print("PCA 2d loss:", loss_2)
print("PCA 3d loss:", loss_3)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for category in y.unique():
    mask = y == category

    axes[0].scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=category
    )

    axes[1].scatter(
        X_pca_sklearn[mask, 0],
        X_pca_sklearn[mask, 1],
        label=category
    )

axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].set_title("Ручной PCA")

axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("Sklearn PCA")

axes[0].legend()
axes[1].legend()

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

for category in y.unique():
    mask = y == category

    ax1.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        X_pca[mask, 2],
        label=category
    )

    ax2.scatter(
        X_pca_sklearn[mask, 0],
        X_pca_sklearn[mask, 1],
        X_pca_sklearn[mask, 2],
        label=category
    )

ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_zlabel("PC3")
ax1.set_title("Ручной PCA")

ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_zlabel("PC3")
ax2.set_title("Sklearn PCA")

ax1.legend()
ax2.legend()

plt.tight_layout()
plt.show()