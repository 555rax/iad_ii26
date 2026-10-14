import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from ucimlrepo import fetch_ucirepo

HIDDEN = [64, 32, 16]
AE_EPOCHS = 30
RBM_EPOCHS = 30
FT_EPOCHS = 60
BATCH = 64
LR_AE = 1e-3
LR_FT = 1e-3
LR_RBM = 0.01
LR_GRBM = 0.002
SEEDS = [0, 1, 2, 3, 4]
MODES = ["без предобучения", "автоэнкодер", "RBM"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_hcv():
    ds = fetch_ucirepo(id=503)
    X = ds.data.features.copy()
    y = ds.data.targets.iloc[:, 0]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median(numeric_only=True))
    return X.values.astype(np.float32), y.values, "numeric"


def load_mushroom():
    ds = fetch_ucirepo(id=73)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df = df.replace("?", np.nan).dropna()
    y = df["poisonous"].values
    X = df.drop(columns=["poisonous"])
    return X, y, "categorical"


def prepare(X, y, kind, seed):
    y = LabelEncoder().fit_transform(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed)
    if kind == "numeric":
        sc = StandardScaler().fit(X_tr)
        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    else:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(X_tr)
        X_tr, X_te = enc.transform(X_tr), enc.transform(X_te)
    return (X_tr.astype(np.float32), X_te.astype(np.float32),
            y_tr.astype(np.int64), y_te.astype(np.int64))


def build_mlp(in_dim, n_classes):
    layers, prev = [], in_dim
    for h in HIDDEN:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


def train_classifier(model, X_tr, y_tr, X_te, y_te, seed):
    torch.manual_seed(seed)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=LR_FT)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                        batch_size=BATCH, shuffle=True)
    Xte, yte = torch.tensor(X_te).to(device), torch.tensor(y_te).to(device)
    hist = {"train_loss": [], "test_loss": []}
    for _ in range(FT_EPOCHS):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        hist["train_loss"].append(tot / len(X_tr))
        model.eval()
        with torch.no_grad():
            hist["test_loss"].append(crit(model(Xte), yte).item())
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1).cpu().numpy()
    return pred, hist


def linear_layers_of(model):
    return [m for m in model if isinstance(m, nn.Linear)][:-1]


def pretrain_autoencoder(model, X_tr, seed):
    torch.manual_seed(seed)
    H = torch.tensor(X_tr).to(device)
    for layer in linear_layers_of(model):
        enc = nn.Sequential(nn.Linear(layer.in_features, layer.out_features), nn.ReLU()).to(device)
        dec = nn.Linear(layer.out_features, layer.in_features).to(device)
        opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR_AE)
        crit = nn.MSELoss()
        loader = DataLoader(TensorDataset(H), batch_size=BATCH, shuffle=True)
        for _ in range(AE_EPOCHS):
            for (xb,) in loader:
                opt.zero_grad()
                crit(dec(enc(xb)), xb).backward()
                opt.step()
        layer.weight.data.copy_(enc[0].weight.data)
        layer.bias.data.copy_(enc[0].bias.data)
        with torch.no_grad():
            H = enc(H)


class RBM:
    def __init__(self, n_vis, n_hid, gaussian_visible=False, lr=0.01):
        self.W = (0.01 * torch.randn(n_vis, n_hid)).to(device)
        self.b = torch.zeros(n_vis, device=device)
        self.c = torch.zeros(n_hid, device=device)
        self.gaussian = gaussian_visible
        self.lr = lr

    def hidden_prob(self, v):
        return torch.sigmoid(v @ self.W + self.c)

    def visible_mean(self, h):
        a = h @ self.W.T + self.b
        return a if self.gaussian else torch.sigmoid(a)

    def fit(self, V, epochs, batch=BATCH):
        loader = DataLoader(TensorDataset(V), batch_size=batch, shuffle=True)
        errors = []
        for _ in range(epochs):
            tot = 0.0
            for (v0,) in loader:
                ph0 = self.hidden_prob(v0)
                h0 = torch.bernoulli(ph0)
                v1 = self.visible_mean(h0)
                ph1 = self.hidden_prob(v1)
                n = v0.shape[0]
                self.W += self.lr * (v0.T @ ph0 - v1.T @ ph1) / n
                self.b += self.lr * (v0 - v1).mean(0)
                self.c += self.lr * (ph0 - ph1).mean(0)
                tot += ((v0 - v1) ** 2).mean().item() * n
            errors.append(tot / len(V))
        return errors


def _is_binary(X):
    return bool(np.isin(X, [0.0, 1.0]).all())


def pretrain_rbm(model, X_tr, seed):
    torch.manual_seed(seed)
    V = torch.tensor(X_tr).to(device)
    errs_all = []
    for k, layer in enumerate(linear_layers_of(model)):
        gauss = (k == 0) and not _is_binary(X_tr)
        rbm = RBM(layer.in_features, layer.out_features,
                  gaussian_visible=gauss, lr=LR_GRBM if gauss else LR_RBM)
        errs_all.append(rbm.fit(V, RBM_EPOCHS))
        layer.weight.data.copy_(rbm.W.T)
        layer.bias.data.copy_(rbm.c)
        with torch.no_grad():
            V = rbm.hidden_prob(V)
    return errs_all


def run_dataset(name, X, y, kind):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    rows, last = [], {}
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = prepare(X, y, kind, seed)
        n_classes = len(np.unique(y_tr))
        for mode in MODES:
            torch.manual_seed(seed)
            model = build_mlp(X_tr.shape[1], n_classes)
            pre = None
            if mode == "автоэнкодер":
                pretrain_autoencoder(model, X_tr, seed)
            elif mode == "RBM":
                pre = pretrain_rbm(model, X_tr, seed)
            pred, hist = train_classifier(model, X_tr, y_tr, X_te, y_te, seed)
            rows.append({"seed": seed, "mode": mode,
                         "accuracy": accuracy_score(y_te, pred),
                         "f1_macro": f1_score(y_te, pred, average="macro"),
                         "f1_weighted": f1_score(y_te, pred, average="weighted")})
            last[mode] = dict(y_te=y_te, pred=pred, hist=hist, pre=pre)

    res = pd.DataFrame(rows)
    res.to_csv(f"lab4_results_{name}.csv", index=False)
    summary = res.groupby("mode")[["accuracy", "f1_macro", "f1_weighted"]].agg(["mean", "std"])
    print(summary.reindex(MODES).round(4).to_string())

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, mode in enumerate(MODES):
        cm = confusion_matrix(last[mode]["y_te"], last[mode]["pred"])
        ax[i].imshow(cm, cmap="Blues")
        ax[i].set_title(f"{name}: {mode}")
        ax[i].set_xlabel("Предсказано")
        ax[i].set_ylabel("Истина")
        for a in range(cm.shape[0]):
            for b in range(cm.shape[1]):
                ax[i].text(b, a, cm[a, b], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(f"lab4_{name}_confusion.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for mode in MODES:
        ax[0].plot(last[mode]["hist"]["test_loss"], label=mode)
    ax[0].set_title(f"{name}: test loss при дообучении")
    ax[0].set_xlabel("Эпоха")
    ax[0].legend()
    ax[0].grid(True)
    for k, e in enumerate(last["RBM"]["pre"]):
        ax[1].plot(e, label=f"RBM слой {k + 1}")
    ax[1].set_title(f"{name}: ошибка реконструкции RBM")
    ax[1].set_xlabel("Эпоха")
    ax[1].set_ylabel("MSE")
    ax[1].legend()
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig(f"lab4_{name}_curves.png", dpi=150)
    plt.close()

    means = summary.reindex(MODES)
    plt.figure(figsize=(7, 4.5))
    xs = np.arange(len(MODES))
    plt.bar(xs - 0.2, means[("accuracy", "mean")], 0.4,
            yerr=means[("accuracy", "std")], label="accuracy")
    plt.bar(xs + 0.2, means[("f1_macro", "mean")], 0.4,
            yerr=means[("f1_macro", "std")], label="macro F1")
    plt.xticks(xs, MODES)
    plt.title(f"{name}: сравнение режимов (среднее ± std, {len(SEEDS)} запусков)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"lab4_{name}_compare.png", dpi=150)
    plt.close()

    print("\nОтчёт по классам (последний seed, RBM):")
    print(classification_report(last["RBM"]["y_te"], last["RBM"]["pred"], zero_division=0))


if __name__ == "__main__":
    X, y, kind = load_hcv()
    run_dataset("HCV", X, y, kind)
    X, y, kind = load_mushroom()
    run_dataset("Mushroom", X, y, kind)