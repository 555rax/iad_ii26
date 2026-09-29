import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


DATA_PATH = "Rice_Cammeo_Osmancik.arff"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MARKERS = ["o", "^"]
COLORS = ["tab:red", "tab:blue"]


def prepare_data(path: str = DATA_PATH):
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    elif path.endswith(".arff"):
        from scipy.io import arff
        raw, _ = arff.loadarff(path)
        df = pd.DataFrame(raw)
        df["Class"] = df["Class"].str.decode("utf-8")
    else:
        df = pd.read_csv(path)

    df = df.fillna(df.mean(numeric_only=True))
    labels, names = pd.factorize(df["Class"])
    data = df.drop(columns=["Class"]).to_numpy(dtype=float)
    data = StandardScaler().fit_transform(data)
    return data, labels, list(names)


def pca_manual(data: np.ndarray, n_components: int):
    data_centered = data - data.mean(axis=0)
    cov_matrix = np.cov(data_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sort_index = np.argsort(-1 * eigenvalues)
    eigenvalues_sorted = eigenvalues[sort_index]
    eigenvectors_sorted = eigenvectors[:, sort_index]
    projected = np.dot(data_centered, eigenvectors_sorted[:, :n_components])
    return projected.real, eigenvalues_sorted.real


def explained_loss(eigenvalues_sorted: np.ndarray, n_components: int) -> float:
    return 100 - eigenvalues_sorted[:n_components].sum() / eigenvalues_sorted.sum() * 100

class Autoencoder(nn.Module):
    def __init__(self, n_features: int, n_latent: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, n_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(data: np.ndarray, n_latent: int, epochs: int = 150,
                      batch_size: int = 64, lr: float = 1e-3):
    x = torch.tensor(data, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True)
    model = Autoencoder(data.shape[1], n_latent)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        if epoch % 50 == 0 or epoch == 1:
            print(f"AE ({n_latent} нейрона), эпоха {epoch}: MSE = {total / len(x):.5f}")

    model.eval()
    with torch.no_grad():
        latent = model.encoder(x).numpy()
        final_mse = criterion(model(x), x).item()
    return latent, final_mse


def run_tsne(data: np.ndarray, n_components: int, perplexity: int):
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                init="pca", random_state=SEED)
    return tsne.fit_transform(data)


def scatter(ax, points, labels, names, title, axis_label):
    is3d = points.shape[1] == 3
    for i, name in enumerate(names):
        m = labels == i
        args = [points[m, j] for j in range(points.shape[1])]
        ax.scatter(*args, c=COLORS[i], marker=MARKERS[i], s=18, alpha=0.7, label=name)
    ax.set_xlabel(f"{axis_label}1")
    ax.set_ylabel(f"{axis_label}2")
    if is3d:
        ax.set_zlabel(f"{axis_label}3")
    ax.set_title(title)
    ax.legend()


def plot_pair(points_2d, points_3d, labels, names, title, axis_label, filename):
    fig = plt.figure(figsize=(13, 5.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    scatter(ax1, points_2d, labels, names, f"{title}, 2D", axis_label)
    scatter(ax2, points_3d, labels, names, f"{title}, 3D", axis_label)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)


def main():
    data, labels, names = prepare_data()
    print(f"Объектов: {data.shape[0]}, признаков: {data.shape[1]}, классы: {names}")


    ae_2d, mse_2 = train_autoencoder(data, n_latent=2)
    ae_3d, mse_3 = train_autoencoder(data, n_latent=3)
    print(f"Итоговая MSE автоэнкодера: 2 нейрона = {mse_2:.5f}, 3 нейрона = {mse_3:.5f}")
    plot_pair(ae_2d, ae_3d, labels, names, "Автоэнкодер", "Z", "autoencoder.png")


    perplexities = [20, 30, 40, 60]
    fig, axes = plt.subplots(1, len(perplexities), figsize=(20, 4.5))
    for ax, p in zip(axes, perplexities):
        scatter(ax, run_tsne(data, 2, p), labels, names, f"t-SNE, perplexity={p}", "T")
    fig.tight_layout()
    fig.savefig("tsne_perplexity.png", dpi=150)

    best_perplexity = 30
    tsne_2d = run_tsne(data, 2, best_perplexity)
    tsne_3d = run_tsne(data, 3, best_perplexity)
    plot_pair(tsne_2d, tsne_3d, labels, names,
              f"t-SNE (perplexity={best_perplexity})", "T", "tsne.png")

    pca_2d, eigvals = pca_manual(data, 2)
    pca_3d, _ = pca_manual(data, 3)
    print(f"Потери PCA: 2 компоненты = {explained_loss(eigvals, 2):.2f}%, "
          f"3 компоненты = {explained_loss(eigvals, 3):.2f}%")
    plot_pair(pca_2d, pca_3d, labels, names, "PCA", "PC", "pca.png")

    plt.show()


if __name__ == "__main__":
    main()