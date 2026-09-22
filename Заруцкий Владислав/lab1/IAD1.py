import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

print("Размер датасета:", df.shape)
print("\nПервые строки:")
print(df.head())
print("\nПропущенные значения:")
print(df.isna().sum())

target_col = "DEATH_EVENT"

if df.isna().sum().sum() > 0:
    df = df.fillna(df.median(numeric_only=True))

y = df[target_col].values
X = df.drop(columns=[target_col]).values

print("\nПризнаки:", df.drop(columns=[target_col]).columns.tolist())
print("Размер X:", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_centered = X_scaled - np.mean(X_scaled, axis=0)

cov_matrix = np.cov(X_centered, rowvar=False)
print("\nРазмер ковариационной матрицы:", cov_matrix.shape)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

eigenvalues = np.real(eigenvalues)
eigenvectors = np.real(eigenvectors)

sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

print("\nСобственные значения (по убыванию):")
print(eigenvalues)

explained_var_ratio_manual = eigenvalues / np.sum(eigenvalues)
print("\nДоли объяснённой дисперсии (вручную):")
print(explained_var_ratio_manual)

W2 = eigenvectors[:, :2]
W3 = eigenvectors[:, :3]

X_pca2_manual = X_centered @ W2
X_pca3_manual = X_centered @ W3

pca_full = PCA()
X_pca_sklearn = pca_full.fit_transform(X_scaled)

print("\nДоли объяснённой дисперсии (sklearn):")
print(pca_full.explained_variance_ratio_)

X_pca2_sklearn = X_pca_sklearn[:, :2]
X_pca3_sklearn = X_pca_sklearn[:, :3]

print("\nПроверка совпадения (2 компоненты):",
      np.allclose(np.abs(X_pca2_manual), np.abs(X_pca2_sklearn), atol=1e-6))

classes = np.unique(y)
colors = ["tab:blue", "tab:red"]
markers = ["o", "s"]

def plot_projection(data2d, title, filename):
    plt.figure(figsize=(8, 6))
    for cls, color, marker in zip(classes, colors, markers):
        mask = y == cls
        plt.scatter(data2d[mask, 0], data2d[mask, 1],
                    c=color, marker=marker, alpha=0.7,
                    label=f"class = {cls}", edgecolor="k", s=45)
    plt.xlabel("Главная компонента 1")
    plt.ylabel("Главная компонента 2")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

plot_projection(X_pca2_manual,
                "PCA (вручную, numpy.linalg.eig): 2 главные компоненты",
                "pca_manual_2d.png")

plot_projection(X_pca2_sklearn,
                "PCA (sklearn): 2 главные компоненты",
                "pca_sklearn_2d.png")

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
for cls, color, marker in zip(classes, colors, markers):
    mask = y == cls
    ax.scatter(X_pca3_manual[mask, 0],
               X_pca3_manual[mask, 1],
               X_pca3_manual[mask, 2],
               c=color, marker=marker, alpha=0.7,
               label=f"class = {cls}", edgecolor="k", s=45)
ax.set_xlabel("ГК 1")
ax.set_ylabel("ГК 2")
ax.set_zlabel("ГК 3")
ax.set_title("PCA (вручную): 3 главные компоненты")
ax.legend()
plt.tight_layout()
plt.savefig("pca_manual_3d.png", dpi=150)
plt.show()

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
for cls, color, marker in zip(classes, colors, markers):
    mask = y == cls
    ax.scatter(X_pca3_sklearn[mask, 0],
               X_pca3_sklearn[mask, 1],
               X_pca3_sklearn[mask, 2],
               c=color, marker=marker, alpha=0.7,
               label=f"class = {cls}", edgecolor="k", s=45)
ax.set_xlabel("ГК 1")
ax.set_ylabel("ГК 2")
ax.set_zlabel("ГК 3")
ax.set_title("PCA (sklearn): 3 главные компоненты")
ax.legend()
plt.tight_layout()
plt.savefig("pca_sklearn_3d.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(np.arange(1, len(eigenvalues) + 1), explained_var_ratio_manual,
        alpha=0.7, label="Доля дисперсии")
plt.plot(np.arange(1, len(eigenvalues) + 1),
         np.cumsum(explained_var_ratio_manual),
         "ro-", label="Накопленная дисперсия")
plt.xlabel("Номер главной компоненты")
plt.ylabel("Доля объяснённой дисперсии")
plt.title("Scree plot")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("scree_plot.png", dpi=150)
plt.show()

total_var = np.sum(eigenvalues)

for k in [2, 3]:
    lost_var = np.sum(eigenvalues[k:])
    loss_ratio = lost_var / total_var
    kept_ratio = 1 - loss_ratio
    print(f"\n=== Проекция на {k} главных компонент ===")
    print(f"Сохранённая дисперсия: {kept_ratio * 100:.2f}%")
    print(f"Потерянная дисперсия:  {loss_ratio * 100:.2f}%")

print("\n=== Оценка по sklearn ===")
for k in [2, 3]:
    kept = np.sum(pca_full.explained_variance_ratio_[:k])
    print(f"{k} компонент(ы): сохранено {kept * 100:.2f}%, "
          f"потеряно {(1 - kept) * 100:.2f}%")