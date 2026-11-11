"""
ЛР 3, вариант 7. Предобучение MLP автоэнкодерами.

Два датасета:
  1) HCV for Egyptian patients — 4 класса гистологической стадии (F1–F4)
  2) Mushrooms из ЛР 2 — съедобный / ядовитый

Схема одна и та же:
  baseline  — MLP со случайными весами, сразу классификация
  pretrained — жадное послойное обучение автоэнкодеров, потом fine-tune
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

HIDDEN = (128, 64, 32, 16)
SEED = 42
SEEDS = (42, 43, 44)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HCV: выборка маленькая, сеть глубокая — даём запас эпох и early stopping.
HCV_CFG = dict(
    ae_epochs=25,
    head_epochs=15,
    ft_epochs=60,
    baseline_epochs=75,
    batch_size=32,
    ae_lr=1e-3,
    head_lr=1e-3,
    ft_lr=3e-4,
    baseline_lr=1e-3,
    weight_decay=1e-4,
    dropout=0.15,
    patience=18,
)

# Грибы: данных больше, задача проще.
MUSH_CFG = dict(
    ae_epochs=15,
    head_epochs=8,
    ft_epochs=35,
    baseline_epochs=43,
    batch_size=64,
    ae_lr=1e-3,
    head_lr=1e-3,
    ft_lr=3e-4,
    baseline_lr=1e-3,
    weight_decay=1e-4,
    dropout=0.2,
    patience=12,
)

MUSHROOM_COLUMNS = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

STAGE_NAMES = ["F1", "F2", "F3", "F4"]
MUSH_NAMES = ["edible", "poisonous"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 140,
    }
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Bundle:
    name: str
    slug: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    feature_names: list[str]
    raw_frame: pd.DataFrame
    cfg: dict
    extra: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def _split_scale(X: np.ndarray, y: np.ndarray, seed: int):
    """70/15/15, стратификация, StandardScaler только по train."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def load_hcv(seed: int = SEED) -> Bundle:
    df = pd.read_csv(DATA_DIR / "HCV-Egy-Data.csv")
    df.columns = df.columns.str.strip()
    target = "Baselinehistological staging"
    y = df[target].to_numpy() - 1  # 1..4 -> 0..3
    X = df.drop(columns=[target]).to_numpy(dtype=np.float32)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = _split_scale(X, y, seed)
    return Bundle(
        name="HCV (Egyptian patients)",
        slug="hcv",
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train.astype(np.int64),
        y_val=y_val.astype(np.int64),
        y_test=y_test.astype(np.int64),
        class_names=STAGE_NAMES,
        feature_names=list(df.drop(columns=[target]).columns),
        raw_frame=df,
        cfg=HCV_CFG,
        extra={"target": target},
    )


def load_mushrooms(seed: int = SEED) -> Bundle:
    df = pd.read_csv(
        DATA_DIR / "agaricus-lepiota.data",
        header=None,
        names=MUSHROOM_COLUMNS,
    )
    y = df["class"].map({"e": 0, "p": 1}).to_numpy(dtype=np.int64)
    X_df = pd.get_dummies(df.drop(columns=["class"]), dtype=np.float32)
    X = X_df.to_numpy(dtype=np.float32)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = _split_scale(X, y, seed)
    return Bundle(
        name="Mushrooms (ЛР 2)",
        slug="mushrooms",
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        class_names=MUSH_NAMES,
        feature_names=list(X_df.columns),
        raw_frame=df,
        cfg=MUSH_CFG,
        extra={"raw_features": 22, "after_ohe": X.shape[1]},
    )


class Autoencoder(nn.Module):
    """Один слойный автоэнкодер: in -> hidden -> in."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder_linear = nn.Linear(in_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, in_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder_linear(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


class MLP(nn.Module):
    """
    Полносвязная сеть: in -> 128 -> 64 -> 32 -> 16 -> классы.
    Скрытые Linear лежат в self.encoders — туда же кладём веса автоэнкодеров.
    """

    def __init__(self, in_dim: int, n_classes: int, dropout: float):
        super().__init__()
        dims = [in_dim, *HIDDEN]
        self.encoders = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(HIDDEN[-1], n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.encoders):
            x = torch.relu(layer(x))
            # На последнем скрытом слое дропаут не вешаем — это уже почти признаки для головы.
            if i < len(self.encoders) - 1:
                x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def _loader(X: np.ndarray, y: np.ndarray | None, batch: int, shuffle: bool) -> DataLoader:
    xt = torch.from_numpy(X)
    if y is None:
        ds = TensorDataset(xt, xt)
    else:
        ds = TensorDataset(xt, torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


@torch.no_grad()
def _predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    logits = model(torch.from_numpy(X).to(DEVICE))
    return logits.argmax(dim=1).cpu().numpy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    }


def train_autoencoder(
    ae: Autoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int,
    lr: float,
    batch: int,
) -> dict:
    ae.to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    train_loader = _loader(X_train, None, batch, True)
    hist = {"train": [], "val": []}

    for _ in range(epochs):
        ae.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            pred = ae(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(xb)
            n += len(xb)
        hist["train"].append(running / n)

        ae.eval()
        with torch.no_grad():
            xv = torch.from_numpy(X_val).to(DEVICE)
            hist["val"].append(float(nn.functional.mse_loss(ae(xv), xv).item()))
    return hist


def greedy_pretrain(bundle: Bundle) -> tuple[list[Autoencoder], list[dict]]:
    """
    Жадно учим каждый скрытый слой как автоэнкодер.
    На вход следующего AE идут активации уже обученного энкодера.
    """
    cfg = bundle.cfg
    current_train = bundle.X_train
    current_val = bundle.X_val
    autos: list[Autoencoder] = []
    histories: list[dict] = []

    for hidden in HIDDEN:
        ae = Autoencoder(current_train.shape[1], hidden)
        hist = train_autoencoder(
            ae,
            current_train,
            current_val,
            epochs=cfg["ae_epochs"],
            lr=cfg["ae_lr"],
            batch=cfg["batch_size"],
        )
        autos.append(ae)
        histories.append(hist)
        with torch.no_grad():
            current_train = (
                ae.encode(torch.from_numpy(current_train).to(DEVICE)).cpu().numpy()
            )
            current_val = ae.encode(torch.from_numpy(current_val).to(DEVICE)).cpu().numpy()
    return autos, histories


def load_encoder_weights(model: MLP, autos: list[Autoencoder]) -> None:
    for linear, ae in zip(model.encoders, autos):
        linear.load_state_dict(ae.encoder_linear.state_dict())


def _epoch_supervised(
    model: MLP,
    loader: DataLoader,
    opt: torch.optim.Optimizer | None,
) -> float:
    train_mode = opt is not None
    model.train(train_mode)
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        if train_mode:
            opt.zero_grad()
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        if train_mode:
            loss.backward()
            opt.step()
        total += float(loss.item()) * len(xb)
        n += len(xb)
    return total / n


def train_supervised(
    model: MLP,
    bundle: Bundle,
    epochs: int,
    lr: float,
    freeze_encoders: bool = False,
    start_hist: dict | None = None,
) -> dict:
    cfg = bundle.cfg
    for p in model.encoders.parameters():
        p.requires_grad = not freeze_encoders
    for p in model.head.parameters():
        p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=cfg["weight_decay"])
    train_loader = _loader(bundle.X_train, bundle.y_train, cfg["batch_size"], True)

    hist = start_hist or {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1 = -1.0
    best_state = None
    wait = 0

    for _ in range(epochs):
        train_loss = _epoch_supervised(model, train_loader, opt)
        val_loss = _epoch_supervised(
            model, _loader(bundle.X_val, bundle.y_val, cfg["batch_size"], False), None
        )
        train_f1 = f1_score(
            bundle.y_train, _predict(model, bundle.X_train), average="macro", zero_division=0
        )
        val_f1 = f1_score(
            bundle.y_val, _predict(model, bundle.X_val), average="macro", zero_division=0
        )
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_f1"].append(float(train_f1))
        hist["val_f1"].append(float(val_f1))

        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


def run_baseline(bundle: Bundle) -> tuple[MLP, dict]:
    model = MLP(bundle.n_features, bundle.n_classes, bundle.cfg["dropout"]).to(DEVICE)
    hist = train_supervised(
        model,
        bundle,
        epochs=bundle.cfg["baseline_epochs"],
        lr=bundle.cfg["baseline_lr"],
    )
    return model, hist


def run_pretrained(bundle: Bundle) -> tuple[MLP, dict, list[dict]]:
    autos, ae_hist = greedy_pretrain(bundle)
    model = MLP(bundle.n_features, bundle.n_classes, bundle.cfg["dropout"]).to(DEVICE)
    load_encoder_weights(model, autos)

    # Сначала крутим только голову, чтобы не развалить уже собранные признаки.
    hist = train_supervised(
        model,
        bundle,
        epochs=bundle.cfg["head_epochs"],
        lr=bundle.cfg["head_lr"],
        freeze_encoders=True,
    )
    # Потом размораживаем всю сеть и доучиваем меньшим шагом.
    hist = train_supervised(
        model,
        bundle,
        epochs=bundle.cfg["ft_epochs"],
        lr=bundle.cfg["ft_lr"],
        freeze_encoders=False,
        start_hist=hist,
    )
    return model, hist, ae_hist


def evaluate_model(model: MLP, bundle: Bundle) -> dict:
    pred = _predict(model, bundle.X_test)
    return _metrics(bundle.y_test, pred, bundle.class_names)


def subset_bundle(bundle: Bundle, frac: float, seed: int) -> Bundle:
    """Оставляет долю размеченных train-объектов. Валидация и тест не трогаем."""
    idx, _, y_keep, _ = train_test_split(
        np.arange(len(bundle.y_train)),
        bundle.y_train,
        train_size=frac,
        random_state=seed,
        stratify=bundle.y_train,
    )
    return Bundle(
        name=bundle.name,
        slug=bundle.slug,
        X_train=bundle.X_train[idx],
        X_val=bundle.X_val,
        X_test=bundle.X_test,
        y_train=bundle.y_train[idx],
        y_val=bundle.y_val,
        y_test=bundle.y_test,
        class_names=bundle.class_names,
        feature_names=bundle.feature_names,
        raw_frame=bundle.raw_frame,
        cfg=bundle.cfg,
        extra=bundle.extra,
    )


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_class_balance(bundle: Bundle, path: Path) -> None:
    if bundle.slug == "mushrooms":
        labels = bundle.raw_frame["class"].map({"e": "edible", "p": "poisonous"})
        values = labels.value_counts().reindex(bundle.class_names)
    else:
        values = (
            bundle.raw_frame["Baselinehistological staging"]
            .value_counts()
            .sort_index()
        )
        values.index = bundle.class_names
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bars = ax.bar(values.index.astype(str), values.to_numpy(), color="#3D5A80")
    ax.set_ylabel("Число объектов")
    ax.set_title(f"{bundle.name}: распределение классов")
    for bar, val in zip(bars, values.to_numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(values) * 0.01, str(int(val)), ha="center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_scale_problem(bundle: Bundle, path: Path) -> None:
    """Показывает, зачем вообще нормализация: RBC и Age живут в разных вселенных."""
    cols = ["Age", "BMI", "WBC", "RBC", "Plat", "RNA Base"]
    fig, axes = plt.subplots(1, len(cols), figsize=(12.5, 3.4))
    for ax, col in zip(axes, cols):
        ax.boxplot(bundle.raw_frame[col], widths=0.5)
        ax.set_title(col)
        ax.set_xticks([])
    fig.suptitle("HCV: сырые признаки в разных масштабах", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_ae_curves(ae_hist: list[dict], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(ae_hist), figsize=(12.8, 3.3), sharey=False)
    if len(ae_hist) == 1:
        axes = [axes]
    dims = [f"{HIDDEN[i]} н." for i in range(len(ae_hist))]
    for ax, hist, dim, i in zip(axes, ae_hist, dims, range(1, 5)):
        ax.plot(hist["train"], label="train MSE")
        ax.plot(hist["val"], label="val MSE")
        ax.set_title(f"AE{i}: → {dim}")
        ax.set_xlabel("Эпоха")
        ax.grid(alpha=0.25)
        if i == 1:
            ax.set_ylabel("MSE")
            ax.legend(fontsize=8)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_learning(base_hist: dict, pre_hist: dict, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    axes[0].plot(base_hist["train_loss"], color="#3D5A80", label="baseline train")
    axes[0].plot(base_hist["val_loss"], color="#3D5A80", ls="--", label="baseline val")
    axes[0].plot(pre_hist["train_loss"], color="#E07A5F", label="pretrained train")
    axes[0].plot(pre_hist["val_loss"], color="#E07A5F", ls="--", label="pretrained val")
    axes[0].set_title("Cross-entropy")
    axes[0].set_xlabel("Эпоха")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(base_hist["train_f1"], color="#3D5A80", label="baseline train")
    axes[1].plot(base_hist["val_f1"], color="#3D5A80", ls="--", label="baseline val")
    axes[1].plot(pre_hist["train_f1"], color="#E07A5F", label="pretrained train")
    axes[1].plot(pre_hist["val_f1"], color="#E07A5F", ls="--", label="pretrained val")
    axes[1].set_title("Macro-F1")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, names: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_compare(rows: list[dict], title: str, path: Path) -> None:
    methods = [r["method"] for r in rows]
    f1s = [r["test"]["f1_macro"] for r in rows]
    accs = [r["test"]["accuracy"] for r in rows]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    w = 0.35
    b1 = ax.bar(x - w / 2, accs, w, label="Accuracy", color="#3D5A80")
    b2 = ax.bar(x + w / 2, f1s, w, label="Macro-F1", color="#E07A5F")
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.3f}",
                ha="center",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _pack_run(method: str, model: MLP, hist: dict, bundle: Bundle, extra: dict | None = None) -> dict:
    rec = {
        "method": method,
        "architecture": [bundle.n_features, *HIDDEN, bundle.n_classes],
        "n_train": int(len(bundle.y_train)),
        "n_val": int(len(bundle.y_val)),
        "n_test": int(len(bundle.y_test)),
        "epochs_ran": len(hist["train_loss"]),
        "history": hist,
        "test": evaluate_model(model, bundle),
        "val_f1_best": float(max(hist["val_f1"])) if hist["val_f1"] else None,
    }
    if extra:
        rec.update(extra)
    return rec


def experiment_one(bundle: Bundle, out: Path, seed: int) -> dict:
    set_seed(seed)
    print(f"\n=== {bundle.name} | seed={seed} | device={DEVICE} ===")
    print(
        f"признаков={bundle.n_features}, классов={bundle.n_classes}, "
        f"train/val/test={len(bundle.y_train)}/{len(bundle.y_val)}/{len(bundle.y_test)}"
    )

    base_model, base_hist = run_baseline(bundle)
    base_rec = _pack_run("baseline", base_model, base_hist, bundle)
    print(
        f"  baseline     acc={base_rec['test']['accuracy']:.4f}  "
        f"macro-F1={base_rec['test']['f1_macro']:.4f}"
    )

    set_seed(seed)
    pre_model, pre_hist, ae_hist = run_pretrained(bundle)
    pre_rec = _pack_run(
        "pretrained",
        pre_model,
        pre_hist,
        bundle,
        extra={"ae_final_val_mse": [h["val"][-1] for h in ae_hist]},
    )
    print(
        f"  pretrained   acc={pre_rec['test']['accuracy']:.4f}  "
        f"macro-F1={pre_rec['test']['f1_macro']:.4f}"
    )

    if seed == SEED:
        torch.save(base_model.state_dict(), out / f"{bundle.slug}_baseline.pt")
        torch.save(pre_model.state_dict(), out / f"{bundle.slug}_pretrained.pt")
        plot_learning(
            base_hist,
            pre_hist,
            f"{bundle.name}: обучение с метками",
            out / f"{bundle.slug}_learning.png",
        )
        plot_ae_curves(
            ae_hist,
            f"{bundle.name}: послойные автоэнкодеры",
            out / f"{bundle.slug}_ae_loss.png",
        )
        plot_confusion(
            np.array(base_rec["test"]["confusion_matrix"]),
            bundle.class_names,
            f"{bundle.name} — baseline",
            out / f"{bundle.slug}_cm_baseline.png",
        )
        plot_confusion(
            np.array(pre_rec["test"]["confusion_matrix"]),
            bundle.class_names,
            f"{bundle.name} — pretrained",
            out / f"{bundle.slug}_cm_pretrained.png",
        )
        plot_compare(
            [base_rec, pre_rec],
            f"{bundle.name}: test-метрики (seed={seed})",
            out / f"{bundle.slug}_compare.png",
        )

    return {"baseline": base_rec, "pretrained": pre_rec}


def limited_labels(full: Bundle, out: Path, frac: float) -> dict:
    """
    Автоэнкодеры видят весь train без меток, классификатор — только долю меток.
    Так предобучение обычно и проявляет себя: меток мало, а структуры в X полно.
    """
    set_seed(SEED)
    labeled = subset_bundle(full, frac, SEED)
    pct = int(round(frac * 100))
    print(
        f"\n=== {full.name}, мало меток: {len(labeled.y_train)} / "
        f"{len(full.y_train)} train ({pct}%) ==="
    )

    base_model, base_hist = run_baseline(labeled)
    base_rec = _pack_run(f"baseline_{pct}pct", base_model, base_hist, labeled)

    set_seed(SEED)
    autos, _ae_hist = greedy_pretrain(full)
    model = MLP(full.n_features, full.n_classes, full.cfg["dropout"]).to(DEVICE)
    load_encoder_weights(model, autos)
    hist = train_supervised(
        model, labeled, labeled.cfg["head_epochs"], labeled.cfg["head_lr"], True
    )
    hist = train_supervised(
        model, labeled, labeled.cfg["ft_epochs"], labeled.cfg["ft_lr"], False, hist
    )
    pre_rec = _pack_run(
        f"pretrained_{pct}pct",
        model,
        hist,
        labeled,
        extra={"ae_trained_on": int(len(full.y_train))},
    )
    print(
        f"  baseline_{pct}   acc={base_rec['test']['accuracy']:.4f}  "
        f"macro-F1={base_rec['test']['f1_macro']:.4f}"
    )
    print(
        f"  pretrained_{pct} acc={pre_rec['test']['accuracy']:.4f}  "
        f"macro-F1={pre_rec['test']['f1_macro']:.4f}"
    )
    plot_learning(
        base_hist,
        hist,
        f"{full.name}, {pct}% меток: обучение классификатора",
        out / f"{full.slug}_learning_{pct}pct.png",
    )
    plot_compare(
        [base_rec, pre_rec],
        f"{full.name}, {pct}% меток: test-метрики",
        out / f"{full.slug}_compare_{pct}pct.png",
    )
    plot_confusion(
        np.array(base_rec["test"]["confusion_matrix"]),
        full.class_names,
        f"{full.name}, {pct}% — baseline",
        out / f"{full.slug}_cm_baseline_{pct}pct.png",
    )
    plot_confusion(
        np.array(pre_rec["test"]["confusion_matrix"]),
        full.class_names,
        f"{full.name}, {pct}% — pretrained",
        out / f"{full.slug}_cm_pretrained_{pct}pct.png",
    )
    return {"baseline": base_rec, "pretrained": pre_rec, "n_labeled": int(len(labeled.y_train))}


def plot_overfit_gaps(payload: dict, path: Path) -> None:
    """Зазор train−val F1: чем больше, тем сильнее сеть просто запомнила train."""
    labels, base_gaps, pre_gaps = [], [], []
    for slug, title in (("hcv", "HCV, 100% меток"), ("mushrooms", "Грибы, 100% меток")):
        run = payload["datasets"][slug]["seed_runs"][0]
        bh, ph = run["baseline_history"], run["pretrained_history"]
        labels.append(title)
        base_gaps.append(bh["train_f1"][-1] - bh["val_f1"][-1])
        pre_gaps.append(ph["train_f1"][-1] - ph["val_f1"][-1])

    for key, title in (
        ("hcv_limited_labels", "HCV, 30% меток"),
        ("mushrooms_limited_labels", "Грибы, 2% меток"),
    ):
        if key not in payload:
            continue
        block = payload[key]
        labels.append(title)
        base_gaps.append(
            block["baseline_history"]["train_f1"][-1] - block["baseline_history"]["val_f1"][-1]
        )
        pre_gaps.append(
            block["pretrained_history"]["train_f1"][-1] - block["pretrained_history"]["val_f1"][-1]
        )

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    w = 0.36
    ax.bar(x - w / 2, base_gaps, w, label="baseline", color="#3D5A80")
    ax.bar(x + w / 2, pre_gaps, w, label="pretrained", color="#E07A5F")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("train F1 − val F1")
    ax.set_title("Переобучение: зазор между train и val")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_architectures(out: Path) -> None:
    from matplotlib.patches import FancyBboxPatch

    def box(ax, xy, w, h, text, color):
        ax.add_patch(
            FancyBboxPatch(
                xy,
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=color,
                edgecolor="#1d1d1d",
                linewidth=1.1,
            )
        )
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center")

    def arrow(ax, x, y1, y2):
        ax.annotate(
            "",
            xy=(x, y2),
            xytext=(x, y1),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
        )

    fig, ax = plt.subplots(figsize=(4.8, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")
    layers = [
        (14.4, "Вход\n28 признаков", "#D9E2EC"),
        (12.0, "Dense 128 + ReLU", "#8E9AAF"),
        (9.6, "Dense 64 + ReLU", "#8E9AAF"),
        (7.2, "Dense 32 + ReLU", "#8E9AAF"),
        (4.8, "Dense 16 + ReLU", "#8E9AAF"),
        (2.4, "Dense 4  (логиты)", "#E07A5F"),
        (0.2, "argmax → F1..F4", "#F2CC8F"),
    ]
    for y, text, color in layers:
        box(ax, (2.2, y), 5.6, 1.7, text, color)
    for i in range(len(layers) - 1):
        arrow(ax, 5, layers[i][0], layers[i + 1][0] + 1.7)
    ax.set_title("MLP для HCV")
    fig.tight_layout()
    fig.savefig(out / "arch_mlp.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 8)
    ax.axis("off")
    for x, name, a, b, c in (
        (0.4, "AE1", "28", "128", "28"),
        (7.6, "AE2", "128", "64", "128"),
        (14.8, "AE3", "64", "32", "64"),
    ):
        ax.text(x + 3.1, 7.4, name, ha="center", fontsize=12, fontweight="bold")
        box(ax, (x + 0.6, 5.2), 5.0, 1.5, f"вход  {a}", "#D9E2EC")
        box(ax, (x + 0.6, 2.9), 5.0, 1.5, f"энкодер  {b}", "#8E9AAF")
        box(ax, (x + 0.6, 0.6), 5.0, 1.5, f"декодер  {c}", "#F2CC8F")
        arrow(ax, x + 3.1, 5.2, 4.4)
        arrow(ax, x + 3.1, 2.9, 2.1)
    ax.set_title("Жадное послойное предобучение (ещё AE4: 32 → 16 → 32)")
    fig.tight_layout()
    fig.savefig(out / "arch_ae.png", dpi=150)
    plt.close(fig)


def analyze_and_plot_raw(out: Path) -> dict:
    hcv = load_hcv(SEED)
    mush = load_mushrooms(SEED)
    draw_architectures(out)
    plot_class_balance(hcv, out / "hcv_classes.png")
    plot_class_balance(mush, out / "mushrooms_classes.png")
    plot_scale_problem(hcv, out / "hcv_scales.png")

    corr = hcv.raw_frame.drop(columns=["Baselinehistological staging"]).corrwith(
        hcv.raw_frame["Baselinehistological staging"]
    )
    top = corr.abs().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = ["#E07A5F" if corr[i] < 0 else "#3D5A80" for i in top.index]
    ax.barh(top.index[::-1], corr[top.index][::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Корреляция Пирсона со стадией")
    ax.set_title("HCV: признаки почти не связаны с целевой меткой")
    fig.tight_layout()
    fig.savefig(out / "hcv_target_corr.png")
    plt.close(fig)

    return {
        "hcv": {
            "n": int(len(hcv.raw_frame)),
            "n_features": hcv.n_features,
            "class_counts": hcv.raw_frame["Baselinehistological staging"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "max_abs_corr": float(corr.abs().max()),
            "mean_abs_corr": float(corr.abs().mean()),
        },
        "mushrooms": {
            "n": int(len(mush.raw_frame)),
            "raw_features": 22,
            "after_ohe": mush.n_features,
            "class_counts": mush.raw_frame["class"].value_counts().to_dict(),
        },
    }


def summarize_seeds(runs: list[dict], method: str) -> dict:
    acc = [r[method]["test"]["accuracy"] for r in runs]
    f1 = [r[method]["test"]["f1_macro"] for r in runs]
    return {
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc, ddof=1) if len(acc) > 1 else 0.0),
        "f1_macro_mean": float(np.mean(f1)),
        "f1_macro_std": float(np.std(f1, ddof=1) if len(f1) > 1 else 0.0),
        "values": {"accuracy": acc, "f1_macro": f1},
    }


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    overview = analyze_and_plot_raw(RESULTS_DIR)

    payload = {
        "student": "Ковальчук Артём",
        "group": "ИИ-26",
        "lab": 3,
        "variant": 7,
        "device": str(DEVICE),
        "hidden": list(HIDDEN),
        "seeds": list(SEEDS),
        "overview": overview,
        "datasets": {},
    }

    for slug, loader in (("hcv", load_hcv), ("mushrooms", load_mushrooms)):
        runs = []
        for seed in SEEDS:
            runs.append(experiment_one(loader(seed), RESULTS_DIR, seed))
        payload["datasets"][slug] = {
            "summary": {
                "baseline": summarize_seeds(runs, "baseline"),
                "pretrained": summarize_seeds(runs, "pretrained"),
            },
            "seed_runs": [],
        }
        for seed, run in zip(SEEDS, runs):
            payload["datasets"][slug]["seed_runs"].append(
                {
                    "seed": seed,
                    "baseline": {k: v for k, v in run["baseline"].items() if k != "history"},
                    "pretrained": {k: v for k, v in run["pretrained"].items() if k != "history"},
                    "baseline_history": run["baseline"]["history"],
                    "pretrained_history": run["pretrained"]["history"],
                }
            )

    hcv_lim = limited_labels(load_hcv(SEED), RESULTS_DIR, 0.30)
    payload["hcv_limited_labels"] = {
        "n_labeled": hcv_lim["n_labeled"],
        "baseline": {k: v for k, v in hcv_lim["baseline"].items() if k != "history"},
        "pretrained": {k: v for k, v in hcv_lim["pretrained"].items() if k != "history"},
        "baseline_history": hcv_lim["baseline"]["history"],
        "pretrained_history": hcv_lim["pretrained"]["history"],
    }

    mush_lim = limited_labels(load_mushrooms(SEED), RESULTS_DIR, 0.02)
    payload["mushrooms_limited_labels"] = {
        "n_labeled": mush_lim["n_labeled"],
        "baseline": {k: v for k, v in mush_lim["baseline"].items() if k != "history"},
        "pretrained": {k: v for k, v in mush_lim["pretrained"].items() if k != "history"},
        "baseline_history": mush_lim["baseline"]["history"],
        "pretrained_history": mush_lim["pretrained"]["history"],
    }

    plot_overfit_gaps(payload, RESULTS_DIR / "overfit_gaps.png")
    _save_json(RESULTS_DIR / "metrics.json", payload)
    print("\nГотово. Картинки и metrics.json лежат в", RESULTS_DIR)
