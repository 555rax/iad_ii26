"""
ЛР 4, вариант 7. Предобучение MLP стеком RBM.

Та же сеть, что в ЛР 3:
  вход → 128 → 64 → 32 → 16 → классы

Три режима на одних и тех же сплитах:
  baseline — случайные веса, сразу классификация
  ae       — жадные автоэнкодеры (ЛР 3), потом fine-tune
  rbm      — стек ограниченных машин Больцмана по лекции, потом fine-tune

Датасеты:
  HCV for Egyptian patients — 4 класса гистологической стадии
  Mushrooms из ЛР 2         — съедобный / ядовитый
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

HCV_CFG = dict(
    ae_epochs=25,
    rbm_epochs=40,
    rbm_lr=1e-2,
    rbm_lr_gaussian=5e-3,
    rbm_start_momentum=0.5,
    rbm_final_momentum=0.9,
    rbm_weight_loss=2e-4,
    rbm_min_error=0.0,
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

MUSH_CFG = dict(
    ae_epochs=15,
    rbm_epochs=25,
    rbm_lr=2e-2,
    rbm_lr_gaussian=8e-3,
    rbm_start_momentum=0.5,
    rbm_final_momentum=0.9,
    rbm_weight_loss=2e-4,
    rbm_min_error=0.0,
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
METHODS = ("baseline", "ae", "rbm")
METHOD_LABELS = {
    "baseline": "без предобучения",
    "ae": "автоэнкодер (ЛР 3)",
    "rbm": "RBM",
}
COLORS = {
    "baseline": "#3D5A80",
    "ae": "#E07A5F",
    "rbm": "#81B29A",
}

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
    y = df[target].to_numpy() - 1
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
    """Полносвязная сеть из ЛР 3: in -> 128 -> 64 -> 32 -> 16 -> классы."""

    def __init__(self, in_dim: int, n_classes: int, dropout: float, hidden_act: str = "relu"):
        super().__init__()
        dims = [in_dim, *HIDDEN]
        self.encoders = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(HIDDEN[-1], n_classes)
        self.hidden_act = hidden_act

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        # RBM в лекции — Logistic; ReLU после порогов −1 душит скрытые нейроны.
        if self.hidden_act == "sigmoid":
            return torch.sigmoid(x)
        return torch.relu(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.encoders):
            x = self._act(layer(x))
            if i < len(self.encoders) - 1:
                x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


# ---------------------------------------------------------------------------
# RBM по исходникам лекции (CD-1, момент, Logistic)
# ---------------------------------------------------------------------------


class Logistic:
    """Сигмоида, как act.Logistic в лекции."""

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class Identity:
    """Линейный видимый слой: гауссовская RBM для непрерывных признаков."""

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x


@dataclass
class PretrainParams:
    """class Pretrain_params из лекции."""

    max_epochs: int
    min_error: float
    batch_size: int
    start_momentum: float
    final_momentum: float
    rates: list[float]
    weight_loss: float


class RBM:
    """
    class RBM из лекции, веса — тензоры PyTorch.

    weights: (n_visible, n_hidden), как np.dot(data, weights) в исходнике.
    """

    def __init__(self, shape: tuple[int, int], act_func: tuple):
        n_vis, n_hid = shape
        self.weights = 0.1 * torch.randn(n_vis, n_hid, device=DEVICE)
        self.vis_biases = torch.zeros(1, n_vis, device=DEVICE)
        self.hid_biases = -1.0 * torch.ones(1, n_hid, device=DEVICE)
        self.act_func = act_func

    def hid_activate(self, data: torch.Tensor) -> torch.Tensor:
        weighted_sum = data @ self.weights + self.hid_biases
        return self.act_func[1].apply(weighted_sum)

    def vis_activate(self, data: torch.Tensor) -> torch.Tensor:
        weighted_sum = data @ self.weights.T + self.vis_biases
        return self.act_func[0].apply(weighted_sum)


class RBMPretrain:
    """class RBM_pretrain: один шаг CD-1 с моментом, как в лекции."""

    def __init__(self, params: PretrainParams):
        self.params = params

    def train(
        self,
        data: torch.Tensor,
        shape: tuple[int, int],
        activ_funcs: tuple,
        rate: float,
    ) -> tuple[RBM, torch.Tensor, list[float]]:
        params = self.params
        batch_size = params.batch_size
        n = data.shape[0]
        rbm = RBM(shape, activ_funcs)

        wu_vh = torch.zeros_like(rbm.weights)
        bu_v = torch.zeros_like(rbm.vis_biases)
        bu_h = torch.zeros_like(rbm.hid_biases)
        output = torch.zeros(n, shape[1], device=DEVICE)
        hist: list[float] = []

        epoch = 0
        is_finish = False
        while not is_finish:
            err = 0.0
            momentum = (
                params.start_momentum if epoch < 5 else params.final_momentum
            )
            perm = torch.randperm(n, device=DEVICE)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                v1 = data[idx]
                bs = v1.shape[0]
                h1 = rbm.hid_activate(v1)
                if epoch == params.max_epochs - 1:
                    output[idx] = h1
                if isinstance(activ_funcs[1], Logistic):
                    h_sampled = (h1 > torch.rand_like(h1)).to(h1.dtype)
                else:
                    h_sampled = h1
                v2 = rbm.vis_activate(h_sampled)
                h2 = rbm.hid_activate(v2)

                wu_vh = wu_vh * momentum + rate * ((v1.T @ h1 - v2.T @ h2) / bs)
                bu_v = bu_v * momentum + (rate / bs) * (
                    v1.sum(0, keepdim=True) - v2.sum(0, keepdim=True)
                )
                bu_h = bu_h * momentum + (rate / bs) * (
                    h1.sum(0, keepdim=True) - h2.sum(0, keepdim=True)
                )
                rbm.weights += wu_vh - rate * params.weight_loss * rbm.weights
                rbm.vis_biases += bu_v
                rbm.hid_biases += bu_h
                err += float(((v2 - v1) ** 2).sum().item())

            hist.append(err / n)
            epoch += 1
            is_finish = epoch >= params.max_epochs or err < params.min_error

        if epoch < params.max_epochs:
            with torch.no_grad():
                output = rbm.hid_activate(data)
        return rbm, output, hist


def pretrain_deep_network(
    data: torch.Tensor,
    params: PretrainParams,
    architecture: list[int],
    activ_funcs: list,
) -> tuple[list[RBM], list[list[float]]]:
    """
    def pretrain_deep_network из лекции.

    architecture = [n_in, 128, 64, 32, 16, n_classes]
    Последний слой (голова) только создаётся, без CD — как last_rbm в лекции.
    """
    current = data
    i = 0
    n_in = current.shape[1]
    rbm_stack: list[RBM] = []
    histories: list[list[float]] = []
    trainer = RBMPretrain(params)

    while i < len(architecture) - 1:
        n_out = architecture[i]
        act_vis = activ_funcs[i]
        act_hid = activ_funcs[i + 1]
        print(f"    Pretrain layer {n_in} - {n_out}")
        rbm, current, hist = trainer.train(
            current,
            (n_in, n_out),
            (act_vis, act_hid),
            params.rates[i],
        )
        rbm_stack.append(rbm)
        histories.append(hist)
        n_in = n_out
        i += 1

    n_out = architecture[i]
    last_rbm = RBM((n_in, n_out), (activ_funcs[i], activ_funcs[i + 1]))
    rbm_stack.append(last_rbm)
    return rbm_stack, histories


def greedy_pretrain_rbm(bundle: Bundle) -> tuple[list[RBM], list[list[float]]]:
    """Стек RBM для скрытых слоёв MLP. Голова не предобучается."""
    cfg = bundle.cfg
    architecture = list(HIDDEN) + [bundle.n_classes]
    n_hidden_layers = len(HIDDEN)
    rates = [cfg["rbm_lr_gaussian"]] + [cfg["rbm_lr"]] * (n_hidden_layers - 1)
    params = PretrainParams(
        max_epochs=cfg["rbm_epochs"],
        min_error=cfg["rbm_min_error"],
        batch_size=cfg["batch_size"],
        start_momentum=cfg["rbm_start_momentum"],
        final_momentum=cfg["rbm_final_momentum"],
        rates=rates,
        weight_loss=cfg["rbm_weight_loss"],
    )
    # Первый видимый слой — Identity (данные после StandardScaler).
    # Дальше скрытые активации уже в (0, 1), Logistic как в лекции.
    activ = [Identity()] + [Logistic()] * (n_hidden_layers + 1)
    data = torch.from_numpy(bundle.X_train).to(DEVICE)
    stack, hist = pretrain_deep_network(data, params, architecture, activ)
    return stack, hist


def load_rbm_weights(model: MLP, stack: list[RBM]) -> None:
    """W лекции — (vis, hid); у nn.Linear weight — (out, in)."""
    for linear, rbm in zip(model.encoders, stack):
        with torch.no_grad():
            linear.weight.copy_(rbm.weights.T.contiguous())
            linear.bias.copy_(rbm.hid_biases.squeeze(0))


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


def greedy_pretrain_ae(bundle: Bundle) -> tuple[list[Autoencoder], list[dict]]:
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


def _finetune(model: MLP, bundle: Bundle, *, rbm: bool = False) -> dict:
    # AE — как в ЛР 3: голова, затем мелкий шаг.
    # RBM: короткая голова и полный шаг, иначе замороженные
    # сигмоидные признаки с порогами −1 схлопывают классификатор в один класс.
    if rbm:
        head_epochs = max(4, bundle.cfg["head_epochs"] // 2)
        ft_epochs = bundle.cfg["baseline_epochs"]
        ft_lr = bundle.cfg["baseline_lr"]
    else:
        head_epochs = bundle.cfg["head_epochs"]
        ft_epochs = bundle.cfg["ft_epochs"]
        ft_lr = bundle.cfg["ft_lr"]
    hist = train_supervised(
        model,
        bundle,
        epochs=head_epochs,
        lr=bundle.cfg["head_lr"],
        freeze_encoders=True,
    )
    return train_supervised(
        model,
        bundle,
        epochs=ft_epochs,
        lr=ft_lr,
        freeze_encoders=False,
        start_hist=hist,
    )


def run_baseline(bundle: Bundle) -> tuple[MLP, dict]:
    model = MLP(bundle.n_features, bundle.n_classes, bundle.cfg["dropout"]).to(DEVICE)
    hist = train_supervised(
        model,
        bundle,
        epochs=bundle.cfg["baseline_epochs"],
        lr=bundle.cfg["baseline_lr"],
    )
    return model, hist


def run_ae(bundle: Bundle) -> tuple[MLP, dict, list[dict]]:
    autos, ae_hist = greedy_pretrain_ae(bundle)
    model = MLP(bundle.n_features, bundle.n_classes, bundle.cfg["dropout"]).to(DEVICE)
    load_encoder_weights(model, autos)
    hist = _finetune(model, bundle)
    return model, hist, ae_hist


def run_rbm(bundle: Bundle) -> tuple[MLP, dict, list[list[float]]]:
    stack, rbm_hist = greedy_pretrain_rbm(bundle)
    model = MLP(
        bundle.n_features,
        bundle.n_classes,
        bundle.cfg["dropout"],
        hidden_act="sigmoid",
    ).to(DEVICE)
    load_rbm_weights(model, stack)
    hist = _finetune(model, bundle, rbm=True)
    return model, hist, rbm_hist


def evaluate_model(model: MLP, bundle: Bundle) -> dict:
    pred = _predict(model, bundle.X_test)
    return _metrics(bundle.y_test, pred, bundle.class_names)


def subset_bundle(bundle: Bundle, frac: float, seed: int) -> Bundle:
    idx, _, _, _ = train_test_split(
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
    for ax, hist, i in zip(axes, ae_hist, range(1, 5)):
        ax.plot(hist["train"], label="train MSE")
        ax.plot(hist["val"], label="val MSE")
        ax.set_title(f"AE{i}: → {HIDDEN[i - 1]} н.")
        ax.set_xlabel("Эпоха")
        ax.grid(alpha=0.25)
        if i == 1:
            ax.set_ylabel("MSE")
            ax.legend(fontsize=8)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_rbm_curves(rbm_hist: list[list[float]], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(rbm_hist), figsize=(12.8, 3.3), sharey=False)
    if len(rbm_hist) == 1:
        axes = [axes]
    for ax, hist, i in zip(axes, rbm_hist, range(1, 5)):
        ax.plot(hist, color="#81B29A")
        ax.set_title(f"RBM{i}: → {HIDDEN[i - 1]} н.")
        ax.set_xlabel("Эпоха")
        ax.grid(alpha=0.25)
        if i == 1:
            ax.set_ylabel("MSE реконструкции")
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_learning(hists: dict[str, dict], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))
    for key, hist in hists.items():
        color = COLORS[key]
        label = METHOD_LABELS[key]
        axes[0].plot(hist["train_loss"], color=color, label=f"{label} train")
        axes[0].plot(hist["val_loss"], color=color, ls="--", label=f"{label} val")
        axes[1].plot(hist["train_f1"], color=color, label=f"{label} train")
        axes[1].plot(hist["val_f1"], color=color, ls="--", label=f"{label} val")
    axes[0].set_title("Cross-entropy")
    axes[0].set_xlabel("Эпоха")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_title("Macro-F1")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
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
    methods = [METHOD_LABELS.get(r["method"], r["method"]) for r in rows]
    f1s = [r["test"]["f1_macro"] for r in rows]
    accs = [r["test"]["accuracy"] for r in rows]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
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
        "train_f1_last": float(hist["train_f1"][-1]) if hist["train_f1"] else None,
        "val_f1_last": float(hist["val_f1"][-1]) if hist["val_f1"] else None,
    }
    if extra:
        rec.update(extra)
    return rec


def experiment_one(bundle: Bundle, out: Path, seed: int) -> dict:
    print(f"\n=== {bundle.name} | seed={seed} | device={DEVICE} ===")
    print(
        f"признаков={bundle.n_features}, классов={bundle.n_classes}, "
        f"train/val/test={len(bundle.y_train)}/{len(bundle.y_val)}/{len(bundle.y_test)}"
    )

    set_seed(seed)
    base_model, base_hist = run_baseline(bundle)
    base_rec = _pack_run("baseline", base_model, base_hist, bundle)
    print(
        f"  baseline  acc={base_rec['test']['accuracy']:.4f}  "
        f"macro-F1={base_rec['test']['f1_macro']:.4f}"
    )

    set_seed(seed)
    ae_model, ae_hist, ae_pre_hist = run_ae(bundle)
    ae_rec = _pack_run(
        "ae",
        ae_model,
        ae_hist,
        bundle,
        extra={"ae_final_val_mse": [h["val"][-1] for h in ae_pre_hist]},
    )
    print(
        f"  ae        acc={ae_rec['test']['accuracy']:.4f}  "
        f"macro-F1={ae_rec['test']['f1_macro']:.4f}"
    )

    set_seed(seed)
    rbm_model, rbm_hist, rbm_pre_hist = run_rbm(bundle)
    rbm_rec = _pack_run(
        "rbm",
        rbm_model,
        rbm_hist,
        bundle,
        extra={"rbm_final_recon": [h[-1] for h in rbm_pre_hist]},
    )
    print(
        f"  rbm       acc={rbm_rec['test']['accuracy']:.4f}  "
        f"macro-F1={rbm_rec['test']['f1_macro']:.4f}"
    )

    if seed == SEED:
        torch.save(base_model.state_dict(), out / f"{bundle.slug}_baseline.pt")
        torch.save(ae_model.state_dict(), out / f"{bundle.slug}_ae.pt")
        torch.save(rbm_model.state_dict(), out / f"{bundle.slug}_rbm.pt")
        plot_learning(
            {"baseline": base_hist, "ae": ae_hist, "rbm": rbm_hist},
            f"{bundle.name}: обучение с метками",
            out / f"{bundle.slug}_learning.png",
        )
        plot_ae_curves(
            ae_pre_hist,
            f"{bundle.name}: послойные автоэнкодеры",
            out / f"{bundle.slug}_ae_loss.png",
        )
        plot_rbm_curves(
            rbm_pre_hist,
            f"{bundle.name}: стек RBM (CD-1)",
            out / f"{bundle.slug}_rbm_loss.png",
        )
        for rec in (base_rec, ae_rec, rbm_rec):
            plot_confusion(
                np.array(rec["test"]["confusion_matrix"]),
                bundle.class_names,
                f"{bundle.name} — {METHOD_LABELS[rec['method']]}",
                out / f"{bundle.slug}_cm_{rec['method']}.png",
            )
        plot_compare(
            [base_rec, ae_rec, rbm_rec],
            f"{bundle.name}: test-метрики (seed={seed})",
            out / f"{bundle.slug}_compare.png",
        )

    return {"baseline": base_rec, "ae": ae_rec, "rbm": rbm_rec}


def limited_labels(full: Bundle, out: Path, frac: float) -> dict:
    set_seed(SEED)
    labeled = subset_bundle(full, frac, SEED)
    pct = int(round(frac * 100))
    print(
        f"\n=== {full.name}, мало меток: {len(labeled.y_train)} / "
        f"{len(full.y_train)} train ({pct}%) ==="
    )

    set_seed(SEED)
    base_model, base_hist = run_baseline(labeled)
    base_rec = _pack_run(f"baseline_{pct}pct", base_model, base_hist, labeled)

    set_seed(SEED)
    autos, _ = greedy_pretrain_ae(full)
    ae_model = MLP(full.n_features, full.n_classes, full.cfg["dropout"]).to(DEVICE)
    load_encoder_weights(ae_model, autos)
    ae_hist = _finetune(ae_model, labeled)
    ae_rec = _pack_run(f"ae_{pct}pct", ae_model, ae_hist, labeled)

    set_seed(SEED)
    stack, _ = greedy_pretrain_rbm(full)
    rbm_model = MLP(
        full.n_features,
        full.n_classes,
        full.cfg["dropout"],
        hidden_act="sigmoid",
    ).to(DEVICE)
    load_rbm_weights(rbm_model, stack)
    rbm_hist = _finetune(rbm_model, labeled, rbm=True)
    rbm_rec = _pack_run(f"rbm_{pct}pct", rbm_model, rbm_hist, labeled)

    for rec in (base_rec, ae_rec, rbm_rec):
        print(
            f"  {rec['method']:<16} acc={rec['test']['accuracy']:.4f}  "
            f"macro-F1={rec['test']['f1_macro']:.4f}"
        )

    plot_learning(
        {"baseline": base_hist, "ae": ae_hist, "rbm": rbm_hist},
        f"{full.name}, {pct}% меток: обучение классификатора",
        out / f"{full.slug}_learning_{pct}pct.png",
    )
    plot_compare(
        [
            {**base_rec, "method": "baseline"},
            {**ae_rec, "method": "ae"},
            {**rbm_rec, "method": "rbm"},
        ],
        f"{full.name}, {pct}% меток: test-метрики",
        out / f"{full.slug}_compare_{pct}pct.png",
    )
    return {
        "baseline": base_rec,
        "ae": ae_rec,
        "rbm": rbm_rec,
        "n_labeled": int(len(labeled.y_train)),
    }


def plot_overfit_gaps(payload: dict, path: Path) -> None:
    labels, gaps = [], {m: [] for m in METHODS}
    for slug, title in (("hcv", "HCV, 100% меток"), ("mushrooms", "Грибы, 100% меток")):
        run = payload["datasets"][slug]["seed_runs"][0]
        labels.append(title)
        for m in METHODS:
            h = run[f"{m}_history"]
            gaps[m].append(h["train_f1"][-1] - h["val_f1"][-1])

    for key, title in (
        ("hcv_limited_labels", "HCV, 30% меток"),
        ("mushrooms_limited_labels", "Грибы, 2% меток"),
    ):
        if key not in payload:
            continue
        block = payload[key]
        labels.append(title)
        for m in METHODS:
            h = block[f"{m}_history"]
            gaps[m].append(h["train_f1"][-1] - h["val_f1"][-1])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    w = 0.24
    for i, m in enumerate(METHODS):
        ax.bar(x + (i - 1) * w, gaps[m], w, label=METHOD_LABELS[m], color=COLORS[m])
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
    ax.set_title("MLP для HCV (та же, что в ЛР 3)")
    fig.tight_layout()
    fig.savefig(out / "arch_mlp.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 8)
    ax.axis("off")
    for x, name, a, b in (
        (0.3, "RBM1", "28 vis", "128 hid"),
        (6.0, "RBM2", "128 vis", "64 hid"),
        (11.7, "RBM3", "64 vis", "32 hid"),
        (17.4, "RBM4", "32 vis", "16 hid"),
    ):
        ax.text(x + 2.5, 7.4, name, ha="center", fontsize=12, fontweight="bold")
        box(ax, (x + 0.3, 4.6), 4.4, 1.8, a, "#D9E2EC")
        box(ax, (x + 0.3, 1.2), 4.4, 1.8, b, "#81B29A")
        arrow(ax, x + 2.5, 4.6, 3.0)
    ax.set_title("Жадный стек RBM (CD-1). Голова 16→классы не предобучается")
    fig.tight_layout()
    fig.savefig(out / "arch_rbm.png", dpi=150)
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


def _strip_hist(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "history"}


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    overview = analyze_and_plot_raw(RESULTS_DIR)

    payload = {
        "student": "Ковальчук Артём",
        "group": "ИИ-26",
        "lab": 4,
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
            "summary": {m: summarize_seeds(runs, m) for m in METHODS},
            "seed_runs": [],
        }
        for seed, run in zip(SEEDS, runs):
            payload["datasets"][slug]["seed_runs"].append(
                {
                    "seed": seed,
                    **{m: _strip_hist(run[m]) for m in METHODS},
                    **{f"{m}_history": run[m]["history"] for m in METHODS},
                }
            )

    hcv_lim = limited_labels(load_hcv(SEED), RESULTS_DIR, 0.30)
    payload["hcv_limited_labels"] = {
        "n_labeled": hcv_lim["n_labeled"],
        **{m: _strip_hist(hcv_lim[m]) for m in METHODS},
        **{f"{m}_history": hcv_lim[m]["history"] for m in METHODS},
    }

    mush_lim = limited_labels(load_mushrooms(SEED), RESULTS_DIR, 0.02)
    payload["mushrooms_limited_labels"] = {
        "n_labeled": mush_lim["n_labeled"],
        **{m: _strip_hist(mush_lim[m]) for m in METHODS},
        **{f"{m}_history": mush_lim[m]["history"] for m in METHODS},
    }

    plot_overfit_gaps(payload, RESULTS_DIR / "overfit_gaps.png")
    _save_json(RESULTS_DIR / "metrics.json", payload)
    print("\nГотово. Картинки и metrics.json лежат в", RESULTS_DIR)
