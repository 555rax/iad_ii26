"""
Лабораторная работа №2. Автоэнкодеры.
Вариант 15: Optical Recognition of Handwritten Digits, целевой признак — последний (tra).

Реализовано:
1. Автоэнкодер с 2 и 3 нейронами в скрытом слое (500 эпох, ReLU, Adam)
2. Визуализация через matplotlib с цветовыми маркерами по классам
3. t-SNE (sklearn.manifold.TSNE) с 2 и 3 компонентами, init='pca',
   сравнение при perplexity ∈ [20, 60]
4. PCA (2 и 3 компоненты) с анализом объяснённой дисперсии
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")


digits = load_digits()
X = digits.data          # (1797, 64)
y = digits.target        # метки 0..9

print(f"Форма данных: {X.shape}")
print(f"Классы: {np.unique(y)}")
print(f"Количество классов: {len(np.unique(y))}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Масштабируем к [0, 1] для Sigmoid на выходе декодера
X_min = X_scaled.min(axis=0)
X_max = X_scaled.max(axis=0)
X_norm = (X_scaled - X_min) / (X_max - X_min + 1e-8)

X_tensor = torch.tensor(X_norm, dtype=torch.float32)



class Autoencoder(nn.Module):
    """Полносвязный автоэнкодер: 64 → 32 → 16 → latent_dim → 16 → 32 → 64."""
    def __init__(self, input_dim=64, latent_dim=2):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(latent_dim, X_tensor, epochs=500,
                      batch_size=64, lr=0.001, print_every=50):

    model = Autoencoder(input_dim=64, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(X_tensor),
                        batch_size=batch_size, shuffle=True)

    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(batch)
            loss = criterion(x_hat, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        avg_loss = epoch_loss / len(X_tensor)
        losses.append(avg_loss)

        if (epoch + 1) % print_every == 0:
            print(f"  Эпоха {epoch+1:3d}/{epochs} | MSE: {avg_loss:.6f}")


    model.eval()
    with torch.no_grad():
        _, Z = model(X_tensor.to(device))
        Z = Z.cpu().numpy()

    return model, Z, losses



def plot_2d(embedding, labels, title, filename,
            xlabel="Компонента 1", ylabel="Компонента 2"):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                          c=labels, cmap="tab10", s=15, alpha=0.7)
    plt.colorbar(scatter, label="Класс (цифра)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_3d(embedding, labels, title, filename,
            xlabel="Компонента 1", ylabel="Компонента 2", zlabel="Компонента 3"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                         c=labels, cmap="tab10", s=15, alpha=0.7)
    fig.colorbar(scatter, label="Класс (цифра)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()



def main():

    print("\n" + "="*60)
    print("1. АВТОЭНКОДЕР (2 нейрона в скрытом слое, 500 эпох)")
    print("="*60)
    start = time.time()
    model_ae2, Z_ae2, losses_ae2 = train_autoencoder(
        latent_dim=2, X_tensor=X_tensor, epochs=500
    )
    print(f"⏱  Время: {time.time()-start:.1f} с")
    print(f"Финальный MSE (AE-2): {losses_ae2[-1]:.6f}")

    plot_2d(Z_ae2, y,
            "Автоэнкодер (2 компоненты) — Optical Digits",
            "ae_2d.png",
            xlabel="AE компонента 1", ylabel="AE компонента 2")


    print("\n" + "="*60)
    print("2. АВТОЭНКОДЕР (3 нейрона в скрытом слое, 500 эпох)")
    print("="*60)
    start = time.time()
    model_ae3, Z_ae3, losses_ae3 = train_autoencoder(
        latent_dim=3, X_tensor=X_tensor, epochs=500
    )
    print(f"⏱  Время: {time.time()-start:.1f} с")
    print(f"Финальный MSE (AE-3): {losses_ae3[-1]:.6f}")

    plot_3d(Z_ae3, y,
            "Автоэнкодер (3 компоненты) — Optical Digits",
            "ae_3d.png",
            xlabel="AE 1", ylabel="AE 2", zlabel="AE 3")


    print("\n" + "="*60)
    print("3. t-SNE — сравнение perplexity (диапазон 20–60)")
    print("="*60)

    perplexities = [20, 30, 40, 50, 60]

    # --- 2D: сетка из 5 графиков ---
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, perp in zip(axes, perplexities):
        print(f"  perplexity={perp} (2D)...", end=" ")
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                    init="pca", max_iter=1000)
        Z = tsne.fit_transform(X_scaled)
        ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=8, alpha=0.7)
        ax.set_title(f"perplexity = {perp}", fontsize=12)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.3)
        print("готово")

    plt.suptitle("Влияние perplexity на t-SNE (Optical Digits, 2D)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("tsne_perplexity_comparison.png", dpi=120, bbox_inches="tight")
    plt.show()

    # --- 3D: сетка из 5 графиков ---
    fig = plt.figure(figsize=(25, 6))
    for i, perp in enumerate(perplexities, 1):
        print(f"  perplexity={perp} (3D)...", end=" ")
        tsne = TSNE(n_components=3, perplexity=perp, random_state=42,
                    init="pca", max_iter=1000)
        Z = tsne.fit_transform(X_scaled)
        ax = fig.add_subplot(1, 5, i, projection="3d")
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2],
                   c=y, cmap="tab10", s=6, alpha=0.7)
        ax.set_title(f"perp = {perp}", fontsize=11)
        ax.set_xlabel("t-SNE 1", fontsize=8)
        ax.set_ylabel("t-SNE 2", fontsize=8)
        ax.set_zlabel("t-SNE 3", fontsize=8)
        print("готово")

    plt.suptitle("Влияние perplexity на t-SNE (Optical Digits, 3D)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig("tsne_perplexity_3d.png", dpi=120, bbox_inches="tight")
    plt.show()

    # --- Финальные визуализации t-SNE при perplexity=30 ---
    print("\nФинальные визуализации t-SNE при perplexity=30:")

    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42,
                   init="pca", max_iter=1000)
    Z_tsne_2d = tsne_2d.fit_transform(X_scaled)
    plot_2d(Z_tsne_2d, y,
            "t-SNE (2 компоненты, perplexity=30) — Optical Digits",
            "tsne_2d.png",
            xlabel="t-SNE 1", ylabel="t-SNE 2")

    tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42,
                   init="pca", max_iter=1000)
    Z_tsne_3d = tsne_3d.fit_transform(X_scaled)
    plot_3d(Z_tsne_3d, y,
            "t-SNE (3 компоненты, perplexity=30) — Optical Digits",
            "tsne_3d.png",
            xlabel="t-SNE 1", ylabel="t-SNE 2", zlabel="t-SNE 3")


    print("\n" + "="*60)
    print("4. PCA")
    print("="*60)

    pca = PCA(n_components=3)
    Z_pca = pca.fit_transform(X_scaled)

    var_ratio = pca.explained_variance_ratio_
    print(f"Объяснённая дисперсия (3 компоненты): {var_ratio.sum()*100:.2f}%")
    print(f"  PC1: {var_ratio[0]*100:.2f}%")
    print(f"  PC2: {var_ratio[1]*100:.2f}%")
    print(f"  PC3: {var_ratio[2]*100:.2f}%")

    # График "каменистой осыпи"
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, 4), var_ratio * 100, alpha=0.7, color="steelblue")
    plt.plot(range(1, 4), np.cumsum(var_ratio) * 100,
             marker="o", color="red", label="Накопленная дисперсия")
    plt.xticks(range(1, 4))
    plt.xlabel("Главная компонента")
    plt.ylabel("Объяснённая дисперсия (%)")
    plt.title("PCA: объяснённая дисперсия по компонентам")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pca_variance.png", dpi=150)
    plt.show()

    plot_2d(Z_pca[:, :2], y,
            "PCA (2 компоненты) — Optical Digits",
            "pca_2d.png",
            xlabel="PC 1", ylabel="PC 2")

    plot_3d(Z_pca, y,
            "PCA (3 компоненты) — Optical Digits",
            "pca_3d.png",
            xlabel="PC 1", ylabel="PC 2", zlabel="PC 3")


    plt.figure(figsize=(10, 5))
    plt.plot(losses_ae2, label="AE-2 (2 нейрона)", linewidth=2)
    plt.plot(losses_ae3, label="AE-3 (3 нейрона)", linewidth=2)
    plt.xlabel("Эпоха")
    plt.ylabel("MSE Loss")
    plt.title("Кривые обучения автоэнкодеров (500 эпох)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ae_losses.png", dpi=150)
    plt.show()


    print("\n" + "="*60)
    print("ИТОГОВЫЕ МЕТРИКИ")
    print("="*60)
    print(f"AE-2:  финальный MSE = {losses_ae2[-1]:.6f}")
    print(f"AE-3:  финальный MSE = {losses_ae3[-1]:.6f}")
    print(f"PCA:   объяснённая дисперсия 3 компонентами = "
          f"{var_ratio.sum()*100:.2f}%")
    print(f"t-SNE: оптимальное perplexity = 30 (выбрано из [20, 60])")
    print("\nПо качеству визуальной кластеризации:")
    print("  t-SNE > Автоэнкодер > PCA")


if __name__ == "__main__":
    main()