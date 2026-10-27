from pathlib import Path
import argparse
import copy
import hashlib
import json
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
COLORS = ["#2463a0", "#d16a28"]

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

class Network(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.hidden = nn.ModuleList(nn.Linear(a, b)
            for a, b in zip(dimensions[:-2], dimensions[1:-1]))
        self.head = nn.Linear(dimensions[-2], dimensions[-1])

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.head(x)  

def prepare_energy():
    df = pd.read_csv(ROOT / "data/energydata_complete.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    assert not df.isna().any().any()
    x = df.drop(columns=["date", "Appliances", "rv1", "rv2"]).copy()
    hour = df.date.dt.hour + df.date.dt.minute / 60
    weekday = df.date.dt.dayofweek
    for name, value, period in [("hour", hour, 24), ("weekday", weekday, 7)]:
        x[name + "_sin"] = np.sin(2 * np.pi * value / period)
        x[name + "_cos"] = np.cos(2 * np.pi * value / period)
    y = df.Appliances.to_numpy(dtype=np.float32)
    n = len(df)
    a, b = int(.70 * n), int(.85 * n)
    indices = [np.arange(a), np.arange(a, b), np.arange(b, n)]
    scaler = StandardScaler().fit(x.iloc[indices[0]])
    xs = [scaler.transform(x.iloc[i]).astype(np.float32) for i in indices]
    sy = StandardScaler().fit(y[indices[0], None])
    ys = [sy.transform(y[i, None]).astype(np.float32) for i in indices]
    metadata = {"shape": list(df.shape), "features": list(x.columns),
        "split_sizes": [len(i) for i in indices], "missing": int(df.isna().sum().sum()),
        "date_ranges": [[str(df.date.iloc[i[0]]), str(df.date.iloc[i[-1]])] for i in indices],
        "y_min": float(y.min()), "y_max": float(y.max()), "y_mean": float(y.mean()),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "target_mean": float(sy.mean_[0]), "target_scale": float(sy.scale_[0])}
    for label, idx in zip(["train", "validation", "test"], indices):
        pd.DataFrame({"row_index": idx}).to_csv(OUT / f"energy_{label}_indices.csv", index=False)
    return dict(name="energy", task="regression", xs=xs, ys=ys,
        truth=y[indices[2]], indices=indices, dimensions=[xs[0].shape[1], 64, 32, 16, 8, 1],
        sy=sy, metadata=metadata, dates=df.date.iloc[indices[2]].astype(str).to_numpy(),
        constant=float(y[indices[0]].mean()))

def prepare_digits():
    raw = pd.read_csv(ROOT / "data/optdigits.tra", header=None)
    assert raw.shape[1] == 65 and not raw.isna().any().any()
    x = raw.iloc[:, :-1].to_numpy(dtype=np.float32) / 16.0
    y = raw.iloc[:, -1].to_numpy(dtype=np.int64)
    itrainval, itest = train_test_split(np.arange(len(y)), test_size=.2,
        stratify=y, random_state=42)
    itrain, ival = train_test_split(itrainval, test_size=.2,
        stratify=y[itrainval], random_state=42)
    indices = [itrain, ival, itest]
    for label, idx in zip(["train", "validation", "test"], indices):
        pd.DataFrame({"row_index": idx}).to_csv(OUT / f"digits_{label}_indices.csv", index=False)
    metadata = {"shape": list(raw.shape), "split_sizes": [len(i) for i in indices],
        "class_counts": np.bincount(y, minlength=10).tolist(),
        "split_class_counts": [np.bincount(y[i], minlength=10).tolist() for i in indices],
        "pixel_min": float(raw.iloc[:, :-1].min().min()),
        "pixel_max": float(raw.iloc[:, :-1].max().max()), "missing": 0}
    return dict(name="digits", task="classification", xs=[x[i] for i in indices],
        ys=[y[i] for i in indices], truth=y[itest], dimensions=[64, 128, 64, 32, 16, 10],
        metadata=metadata, indices=indices, images=x[itest])

def batches(n, generator, size=128):
    return torch.randperm(n, generator=generator).split(size)

def pretrain(model, x, seed, epochs=50):
    """Жадное послойное предобучение всех четырёх скрытых слоёв."""
    h = x.clone()
    history = []
    for number, layer in enumerate(model.hidden, 1):
        seed_all(seed + 1000 + number)
        encoder = copy.deepcopy(layer)
        decoder = nn.Linear(layer.out_features, layer.in_features)
        ae = nn.Sequential(encoder, nn.ReLU(), decoder)
        optimizer = torch.optim.Adam(ae.parameters(), lr=.001)
        generator = torch.Generator().manual_seed(seed + number)
        for epoch in range(1, epochs + 1):
            total = 0.
            for ix in batches(len(h), generator):
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(ae(h[ix]), h[ix])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(ix)
            history.append({"layer": number, "epoch": epoch, "mse": total / len(h)})
        layer.load_state_dict(encoder.state_dict())
        with torch.no_grad():
            h = torch.relu(layer(h)).detach()
    return history

def supervised(model, x, y, xv, yv, seed, classification, head_only=False):
    criterion = nn.CrossEntropyLoss() if classification else nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=.001)
    generator = torch.Generator().manual_seed(seed + 5000)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state, history = None, []
    for epoch in range(1, (10 if head_only else 100) + 1):
        model.train()
        total = 0.
        for ix in batches(len(x), generator):
            optimizer.zero_grad()
            loss = criterion(model(x[ix]), y[ix])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ix)
        model.eval()
        with torch.no_grad():
            val = criterion(model(xv), yv).item()
        history.append({"epoch": epoch, "train_loss": total / len(x), "val_loss": val})
        if val < best_loss - 1e-6:
            best_loss, best_epoch, stale = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if not head_only and stale >= 15:
            break
    if not head_only:
        model.load_state_dict(best_state)
    return history, best_epoch

def evaluate(task, truth, prediction):
    if task == "regression":
        assert np.all(truth > 0), "MAPE требует ненулевых значений y"
        return {"MAE": float(mean_absolute_error(truth, prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
            "MAPE_percent": float(100 * mean_absolute_percentage_error(truth, prediction)),
            "R2": float(r2_score(truth, prediction))}
    return {"Accuracy": float(accuracy_score(truth, prediction)),
        "Precision_macro": float(precision_score(truth, prediction, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(truth, prediction, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "F1_weighted": float(f1_score(truth, prediction, average="weighted", zero_division=0))}

def run_dataset(data, seeds):
    rows, histories, predictions = [], {}, {}
    xs = [torch.from_numpy(x) for x in data["xs"]]
    ys = [torch.from_numpy(y) for y in data["ys"]]
    classification = data["task"] == "classification"
    for seed in seeds:
        seed_all(seed)
        initial = Network(data["dimensions"])
        for mode in ["baseline", "pretrained"]:
            model = copy.deepcopy(initial)
            started = time.perf_counter()
            if mode == "pretrained":
                ae_history = pretrain(model, xs[0], seed)
                pd.DataFrame(ae_history).to_csv(OUT / f"{data['name']}_{seed}_ae.csv", index=False)
                for param in model.hidden.parameters():
                    param.requires_grad_(False)
                warm, _ = supervised(model, xs[0], ys[0], xs[1], ys[1], seed,
                    classification, head_only=True)
                pd.DataFrame(warm).to_csv(OUT / f"{data['name']}_{seed}_head.csv", index=False)
                for param in model.hidden.parameters():
                    param.requires_grad_(True)
            hist, best_epoch = supervised(model, xs[0], ys[0], xs[1], ys[1], seed, classification)
            seconds = time.perf_counter() - started
            model.eval()
            with torch.no_grad():
                raw = model(xs[2]).numpy()
            pred = raw.argmax(axis=1) if classification else data["sy"].inverse_transform(raw).ravel()
            row = {"dataset": data["name"], "seed": seed, "mode": mode,
                **evaluate(data["task"], data["truth"], pred),
                "best_epoch": best_epoch, "epochs_run": len(hist), "seconds": seconds}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            pd.DataFrame(hist).to_csv(OUT / f"{data['name']}_{seed}_{mode}_history.csv", index=False)
            pd.DataFrame({"source_row": data["indices"][2], "actual": data["truth"],
                "predicted": pred}).to_csv(OUT / f"{data['name']}_{seed}_{mode}_predictions.csv", index=False)
            if seed == seeds[0]:
                histories[mode], predictions[mode] = hist, pred
                torch.save(model.state_dict(), OUT / f"{data['name']}_{mode}.pt")
                if classification:
                    cm = confusion_matrix(data["truth"], pred, labels=np.arange(10))
                    pd.DataFrame(cm, index=range(10), columns=range(10)).to_csv(OUT / f"digits_{mode}_confusion.csv")
                    report = classification_report(data["truth"], pred, output_dict=True, zero_division=0)
                    pd.DataFrame(report).T.to_csv(OUT / f"digits_{mode}_classification_report.csv")
    plot_results(data, histories, predictions, seeds[0])
    return rows

def plot_results(data, histories, predictions, seed):
    labels = {"baseline": "Без предобучения", "pretrained": "С предобучением"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for color, (mode, history) in zip(COLORS, histories.items()):
        for ax, field, title in zip(axes, ["train_loss", "val_loss"], ["Обучающая часть", "Валидация"]):
            ax.plot([h["epoch"] for h in history], [h[field] for h in history], color=color, label=labels[mode])
            ax.set(xlabel="Эпоха дообучения", ylabel="Cross-entropy" if data["task"] == "classification" else "MSE стандартизованной цели", title=title)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.savefig(OUT / f"{data['name']}_learning.png", dpi=180)
    plt.close(fig)
    if data["task"] == "regression":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, color, (mode, pred) in zip(axes, COLORS, predictions.items()):
            ax.scatter(data["truth"], pred, s=6, alpha=.25, color=color)
            ax.plot([0, 1100], [0, 1100], "--", color="gray", linewidth=1)
            ax.set(xlabel="Факт, Вт·ч", ylabel="Прогноз, Вт·ч", title=labels[mode])
            ax.grid(alpha=.2)
        fig.savefig(OUT / "energy_prediction.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
        ax.plot(data["truth"][:432], label="Факт", color="#222222", linewidth=1)
        for color, (mode, pred) in zip(COLORS, predictions.items()):
            ax.plot(pred[:432], label=labels[mode], color=color, linewidth=1)
        ax.set(xlabel="Шаг 10 минут от начала теста", ylabel="Appliances, Вт·ч", title="Первые 72 часа тестовой части")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.savefig(OUT / "energy_timeline.png", dpi=180); plt.close(fig)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
        for ax, (mode, pred) in zip(axes, predictions.items()):
            cm = confusion_matrix(data["truth"], pred, labels=np.arange(10))
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(axis=1).max())
            for i in range(10):
                for j in range(10):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 40 else "black")
            ax.set(xticks=range(10), yticks=range(10), xlabel="Прогноз", ylabel="Истинный класс", title=labels[mode])
        fig.savefig(OUT / "digits_confusion.png", dpi=180); plt.close(fig)
        fig, axes = plt.subplots(2, 5, figsize=(8, 3.5), constrained_layout=True)
        for digit, ax in enumerate(axes.flat):
            index = np.flatnonzero(data["truth"] == digit)[0]
            ax.imshow(data["images"][index].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(str(digit)); ax.axis("off")
        fig.savefig(OUT / "digits_examples.png", dpi=180); plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    energy, digits = prepare_energy(), prepare_digits()
    metadata = {"python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__, "seeds": args.seeds, "split_seed": 42,
        "energy": energy["metadata"], "digits": digits["metadata"],
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "data").iterdir()},
        "energy_constant": evaluate("regression", energy["truth"], np.full_like(energy["truth"], energy["constant"]))}
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = run_dataset(energy, args.seeds) + run_dataset(digits, args.seeds)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "metrics_all.csv", index=False)
    summary = table.groupby(["dataset", "mode"]).agg({c: ["mean", "std"] for c in table.select_dtypes("number").columns if c != "seed"})
    summary.to_csv(OUT / "metrics_summary.csv")
    print("\nРезультаты сохранены:", OUT, flush=True)

if __name__ == "__main__":
    main()
