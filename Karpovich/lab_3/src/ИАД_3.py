import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CASP_PATH = "CASP.csv"
OUT_DIR = "results"
SHOW_PLOTS = True

SEEDS = [0, 1, 2, 3, 4]
BATCH = 256

AE_EPOCHS = 30
AE_LR = 1e-3
HEAD_EPOCHS = 30

FT_LR = 1e-3

MAPE_MIN_Y = 1.0

DATASETS = {
    "CASP (RMSD, регрессия)": dict(task="regression", hidden=[64, 32, 16, 8], batch=256, epochs=60, fracs=[1.0, 0.05]),
    "WDBC (рак груди, классификация)": dict(task="classification", hidden=[64, 32, 16, 8], batch=32, epochs=100, fracs=[1.0, 0.25]),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_casp():
    if os.path.exists(CASP_PATH):
        df = pd.read_csv(CASP_PATH)
    else:
        try:
            from ucimlrepo import fetch_ucirepo
            ds = fetch_ucirepo(id=265)
            df = pd.concat([ds.data.targets, ds.data.features], axis=1)
        except Exception as e:
            raise FileNotFoundError(
                f"Не найден {CASP_PATH}. Скачайте датасет с "
                "https://archive.ics.uci.edu/dataset/265 и положите CASP.csv рядом со скриптом."
            ) from e
    X = df.drop(columns=["RMSD"]).values.astype(np.float32)
    y = df["RMSD"].values.astype(np.float32)
    return X, y, None


def load_wdbc():
    data = load_breast_cancer()
    return data.data.astype(np.float32), data.target.astype(np.float32), list(data.target_names)


class MLP(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def features(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return x

    def forward(self, x):
        return self.layers[-1](self.features(x))


def batches(n, bs, gen):
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, bs):
        yield perm[i:i + bs].to(DEVICE)


def sup_loss(task, out, target):
    out = out.squeeze(1)
    if task == "regression":
        return F.mse_loss(out, target)
    return F.binary_cross_entropy_with_logits(out, target)


def pretrain_with_autoencoders(model, X, gen):
    ae_history, cur = [], X
    for k, layer in enumerate(model.layers[:-1]):
        decoder = nn.Linear(layer.out_features, layer.in_features).to(DEVICE)
        opt = torch.optim.Adam(list(layer.parameters()) + list(decoder.parameters()), lr=AE_LR)
        losses = []
        for _ in range(AE_EPOCHS):
            total = 0.0
            for idx in batches(len(cur), BATCH, gen):
                xb = cur[idx]
                rec = decoder(torch.relu(layer(xb)))
                loss = F.mse_loss(rec, xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            losses.append(total / len(cur))
        ae_history.append(losses)
        with torch.no_grad():
            cur = torch.relu(layer(cur))
    return ae_history


def train_last_layer(model, X, y, task, gen):
    with torch.no_grad():
        feats = model.features(X)
    head = model.layers[-1]
    opt = torch.optim.Adam(head.parameters(), lr=FT_LR)
    for _ in range(HEAD_EPOCHS):
        for idx in batches(len(feats), BATCH, gen):
            loss = sup_loss(task, head(feats[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def predict_raw(model, X):
    model.eval()
    out = model(X).squeeze(1).cpu().numpy()
    model.train()
    return out


def train_supervised(model, Xtr, ytr, Xte, yte, task, epochs, gen):
    opt = torch.optim.Adam(model.parameters(), lr=FT_LR)
    hist = {"train": [], "test": []}
    for _ in range(epochs):
        total = 0.0
        for idx in batches(len(Xtr), BATCH, gen):
            loss = sup_loss(task, model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        hist["train"].append(total / len(Xtr))
        with torch.no_grad():
            hist["test"].append(sup_loss(task, model(Xte), yte).item())
    return hist


def regression_metrics(y_true, y_pred):
    mask = y_true >= MAPE_MIN_Y
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
            "MAPE, %": mape}


def classification_metrics(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    return {"Accuracy": accuracy_score(y_true, pred),
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred, zero_division=0),
            "F1": f1_score(y_true, pred, zero_division=0)}


def evaluate(model, X, y_true, task, y_scaler):
    raw = predict_raw(model, X)
    if task == "regression":
        pred = y_scaler.inverse_transform(raw.reshape(-1, 1)).ravel()
        return regression_metrics(y_true, pred), pred
    prob = 1 / (1 + np.exp(-np.clip(raw, -50, 50)))
    return classification_metrics(y_true, prob), prob


def run_seed(X_all, y_all, task, hidden, epochs, frac, seed):
    strat = y_all if task == "classification" else None
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, random_state=seed, stratify=strat)
    if frac < 1.0:
        keep = np.random.RandomState(seed).permutation(len(Xtr))[:max(int(len(Xtr) * frac), 20)]
        Xtr, ytr = Xtr[keep], ytr[keep]

    x_scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = x_scaler.transform(Xtr), x_scaler.transform(Xte)
    y_scaler = None
    ytr_s, yte_s = ytr, yte
    if task == "regression":
        y_scaler = StandardScaler().fit(ytr.reshape(-1, 1))
        ytr_s = y_scaler.transform(ytr.reshape(-1, 1)).ravel()
        yte_s = y_scaler.transform(yte.reshape(-1, 1)).ravel()

    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)
    Xtr_t, Xte_t, ytr_t, yte_t = t(Xtr_s), t(Xte_s), t(ytr_s), t(yte_s)

    dims = [X_all.shape[1]] + hidden + [1]
    torch.manual_seed(seed)
    base = MLP(dims).to(DEVICE)
    pre = copy.deepcopy(base)

    hist_base = train_supervised(base, Xtr_t, ytr_t, Xte_t, yte_t, task, epochs, torch.Generator().manual_seed(seed))
    m_base, out_base = evaluate(base, Xte_t, yte, task, y_scaler)

    gen = torch.Generator().manual_seed(seed)
    ae_hist = pretrain_with_autoencoders(pre, Xtr_t, gen)
    train_last_layer(pre, Xtr_t, ytr_t, task, gen)
    m_only, _ = evaluate(pre, Xte_t, yte, task, y_scaler)
    hist_pre = train_supervised(pre, Xtr_t, ytr_t, Xte_t, yte_t, task, epochs, gen)
    m_pre, out_pre = evaluate(pre, Xte_t, yte, task, y_scaler)

    return dict(metrics={"Без предобучения": m_base, "Только предобучение (без дообучения)": m_only,
                         "С предобучением + дообучение": m_pre},
                hist_base=hist_base, hist_pre=hist_pre, ae_hist=ae_hist,
                y_true=yte, out_base=out_base, out_pre=out_pre)


def make_figure(name, frac, task, runs, class_names, path):
    first = runs[0]
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    ax = axes.ravel()

    for k, losses in enumerate(first["ae_hist"]):
        ax[0].plot(losses, label=f"Слой {k + 1}")
    ax[0].set_title("Предобучение: потери реконструкции (MSE)")
    ax[0].set_xlabel("Эпоха")
    ax[0].set_ylabel("MSE")
    ax[0].legend(title="Автоэнкодер")
    ax[0].grid(alpha=0.3)

    for key, label in (("hist_base", "Без предобучения"), ("hist_pre", "С предобучением")):
        ax[1].plot(np.mean([r[key]["test"] for r in runs], axis=0), label=label)
    ax[1].set_title("Ошибка на тесте (среднее по запускам)")
    ax[1].set_xlabel("Эпоха")
    ax[1].set_ylabel("MSE (норм.)" if task == "regression" else "BCE")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    for a, key, title in ((ax[2], "out_base", "Без предобучения"), (ax[3], "out_pre", "С предобучением")):
        if task == "regression":
            n = min(3000, len(first["y_true"]))
            a.scatter(first["y_true"][:n], first[key][:n], s=5, alpha=0.4)
            lim = [0, max(first["y_true"].max(), first[key].max())]
            a.plot(lim, lim, "r--")
            a.set_xlabel("Истинный RMSD")
            a.set_ylabel("Предсказанный RMSD")
            a.set_title(f"{title}: прогноз и факт")
        else:
            cm = confusion_matrix(first["y_true"], (first[key] >= 0.5).astype(int))
            ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=a, colorbar=False)
            a.set_xlabel("Предсказанный класс")
            a.set_ylabel("Истинный класс")
            a.set_title(f"{title}: матрица ошибок")
    fig.suptitle(f"{name}\nдоля обучающей выборки = {frac}", fontsize=14)
    fig.savefig(path, dpi=150)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Устройство: {DEVICE}")
    loaders = {"CASP (RMSD, регрессия)": load_casp, "WDBC (рак груди, классификация)": load_wdbc}
    rows = []

    for name, cfg in DATASETS.items():
        X, y, class_names = loaders[name]()
        task = cfg["task"]
        global BATCH
        BATCH = cfg["batch"]
        print(f"\n{'=' * 70}\n{name}: {X.shape[0]} объектов, {X.shape[1]} признаков")
        print(f"Архитектура: {[X.shape[1]] + cfg['hidden'] + [1]}")
        for frac in cfg["fracs"]:
            runs = []
            t0 = time.time()
            for seed in SEEDS:
                r = run_seed(X, y, task, cfg["hidden"], cfg["epochs"], frac, seed)
                runs.append(r)
                for method, m in r["metrics"].items():
                    rows.append({"Датасет": name, "Доля train": frac, "Метод": method, "seed": seed, **m})
                print(f"  доля={frac}, seed={seed} готово ({time.time() - t0:.0f} c)")
            make_figure(name, frac, task, runs, class_names,
                        os.path.join(OUT_DIR, f"{name.split()[0]}_frac{frac}.png"))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "all_runs.csv"), index=False, encoding="utf-8-sig")
    metric_cols = [c for c in df.columns if c not in ("Датасет", "Доля train", "Метод", "seed")]
    summary = df.groupby(["Датасет", "Доля train", "Метод"], sort=False)[metric_cols].agg(["mean", "std"])
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), encoding="utf-8-sig")

    pd.set_option("display.width", 250, "display.max_columns", 50)
    print("\n" + "=" * 70 + "\nИТОГОВЫЕ РЕЗУЛЬТАТЫ (mean по запускам; std в summary.csv)\n" + "=" * 70)
    for (ds, frac), g in df.groupby(["Датасет", "Доля train"], sort=False):
        print(f"\n{ds}, доля обучающей выборки = {frac}")
        cols = [c for c in metric_cols if g[c].notna().any()]
        print(g.groupby("Метод", sort=False)[cols].mean().round(4).to_string())

    print(f"\nГрафики и таблицы сохранены в папку '{OUT_DIR}'")
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()