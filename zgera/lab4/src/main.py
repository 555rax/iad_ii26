import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))

FEATURES = ["LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", "ALTV",
            "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean",
            "Median", "Variance", "Tendency"]
TARGET = "NSP"


def load_ctg(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx"):
        try:
            raw = pd.read_excel(path, sheet_name="Data", header=None)
        except ImportError as e:
            need = "xlrd" if ext == ".xls" else "openpyxl"
            raise SystemExit(f"Для чтения {ext} нужен пакет {need}. Установите: "
                             f"pip install {need}   (или откройте файл в Excel, "
                             f"сохраните как .xlsx)") from e
        hdr = None
        for i in range(min(len(raw), 10)):
            if TARGET in [str(v).strip() for v in raw.iloc[i].values]:
                hdr = i
                break
        if hdr is None:
            raise ValueError("В листе 'Data' не найдена строка заголовка с колонкой NSP")
        df = raw.iloc[hdr + 1:].copy()
        df.columns = [str(c).strip() for c in raw.iloc[hdr].values]
    else:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]

    dup = df.columns[df.columns.duplicated()].tolist()
    if dup:
        print(f"Повторяющиеся колонки {sorted(set(dup))}: оставлено первое вхождение")
        df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"В файле нет колонок: {missing}. Найдены: {list(df.columns)}")

    df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df.dropna(subset=FEATURES)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int) - 1
    assert set(np.unique(y)) <= {0, 1, 2}, "NSP должен принимать значения 1/2/3"
    return X, y


ACTS = {
    "relu": (lambda z: np.maximum(z, 0.0), lambda a: (a > 0).astype(a.dtype)),
    "tanh": (np.tanh, lambda a: 1.0 - a * a),
    "sigmoid": (lambda z: 1.0 / (1.0 + np.exp(-z)), lambda a: a * (1.0 - a)),
}


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes, act, task, rng):
        self.sizes, self.act, self.task = list(sizes), act, task
        self.f, self.df = ACTS[act]
        self.W, self.b = [], []
        last = len(sizes) - 2
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            if act == "relu" and i < last:
                W = rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out))
            else:
                lim = np.sqrt(6.0 / (n_in + n_out))
                W = rng.uniform(-lim, lim, (n_in, n_out))
            self.W.append(W)
            self.b.append(np.zeros(n_out))

    def forward(self, X):
        A = [X]
        L = len(self.W)
        for i in range(L):
            Z = A[-1] @ self.W[i] + self.b[i]
            if i < L - 1:
                A.append(self.f(Z))
            else:
                A.append(softmax(Z) if self.task == "clf" else Z)
        return A

    def loss(self, X, Y):
        out = self.forward(X)[-1]
        if self.task == "clf":
            return -np.mean(np.log(out[np.arange(len(X)), Y] + 1e-12))
        return np.mean((out - Y) ** 2)

    def grads(self, X, Y, l2):
        A = self.forward(X)
        n, out = len(X), A[-1]
        if self.task == "clf":
            delta = out.copy()
            delta[np.arange(n), Y] -= 1.0
            delta /= n
        else:
            delta = 2.0 * (out - Y) / (n * out.shape[1])
        L = len(self.W)
        gW, gb = [None] * L, [None] * L
        for i in reversed(range(L)):
            gW[i] = A[i].T @ delta + l2 * self.W[i]
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self.df(A[i])
        return gW, gb

    def predict(self, X):
        return self.forward(X)[-1].argmax(axis=1)


def fit(net, X, Y, epochs, lr, batch, l2, rng, eval_fn=None):
    params = net.W + net.b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    n, hist = len(X), []
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            bi = idx[s:s + batch]
            gW, gb = net.grads(X[bi], Y[bi], l2)
            t += 1
            for p, g, mm, vv in zip(params, gW + gb, m, v):
                mm *= b1
                mm += (1 - b1) * g
                vv *= b2
                vv += (1 - b2) * g * g
                p -= lr * (mm / (1 - b1 ** t)) / (np.sqrt(vv / (1 - b2 ** t)) + eps)
        if eval_fn is not None:
            hist.append(eval_fn(net))
    return hist


def pretrain_autoencoders(Xtr, Xva, hidden, act, epochs, lr, batch, l2, noise, seed):
    f = ACTS[act][0]
    Htr, Hva = Xtr, Xva
    enc, curves = [], []
    for k, h in enumerate(hidden):
        rng = np.random.default_rng(seed * 100 + k)
        ae = MLP([Htr.shape[1], h, Htr.shape[1]], act, "ae", rng)
        sd = Htr.std()

        def ev(net, Htr=Htr, Hva=Hva):
            return {"train": net.loss(Htr, Htr), "val": net.loss(Hva, Hva)}

        if noise > 0:
            hist = []
            for _ in range(epochs):
                Xn = Htr + rng.normal(0.0, noise * sd, Htr.shape)
                fit(ae, Xn, Htr, 1, lr, batch, l2, rng)
                hist.append(ev(ae))
        else:
            hist = fit(ae, Htr, Htr, epochs, lr, batch, l2, rng, ev)
        enc.append((ae.W[0].copy(), ae.b[0].copy()))
        curves.append(hist)
        Htr = f(Htr @ ae.W[0] + ae.b[0])
        Hva = f(Hva @ ae.W[0] + ae.b[0])
    return enc, curves


MODES = ["scratch", "ae", "rbm"]
LBL = {"scratch": "Без предобучения", "ae": "Автоэнкодеры", "rbm": "RBM"}
COL = {"scratch": "#2a78d6", "ae": "#eb6834", "rbm": "#2f9e62"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class RBM:
    def __init__(self, n_vis, n_hid, visible, rng):
        self.visible, self.rng = visible, rng
        self.W = rng.normal(0.0, 0.01, (n_vis, n_hid))
        self.a = np.zeros(n_vis)
        self.b = np.zeros(n_hid)

    def p_h(self, v):
        return sigmoid(v @ self.W + self.b)

    def p_v(self, h):
        z = h @ self.W.T + self.a
        return z if self.visible == "gauss" else sigmoid(z)

    def recon_error(self, v):
        return float(np.mean((self.p_v(self.p_h(v)) - v) ** 2))

    def fit(self, V, Vva, epochs, lr, batch, momentum=0.5, final_momentum=0.9,
            wd=1e-4, k=1):
        n = len(V)
        dW = np.zeros_like(self.W)
        da = np.zeros_like(self.a)
        db = np.zeros_like(self.b)
        hist = []
        for ep in range(epochs):
            mom = momentum if ep < 5 else final_momentum
            idx = self.rng.permutation(n)
            for s in range(0, n, batch):
                v0 = V[idx[s:s + batch]]
                m = len(v0)
                ph0 = self.p_h(v0)
                h = (self.rng.random(ph0.shape) < ph0).astype(float)
                for _ in range(k):
                    pv = self.p_v(h)
                    vk = pv
                    phk = self.p_h(vk)
                    h = (self.rng.random(phk.shape) < phk).astype(float)
                gW = (v0.T @ ph0 - vk.T @ phk) / m - wd * self.W
                ga = (v0 - vk).mean(0)
                gb = (ph0 - phk).mean(0)
                dW = mom * dW + lr * gW
                da = mom * da + lr * ga
                db = mom * db + lr * gb
                self.W += dW
                self.a += da
                self.b += db
            hist.append({"train": self.recon_error(V), "val": self.recon_error(Vva)})
        return hist


def pretrain_rbms(Xtr, Xva, hidden, visible, epochs, lr, batch, seed, k=1):
    Htr, Hva = Xtr, Xva
    enc, curves, rbms = [], [], []
    for i, h in enumerate(hidden):
        rng = np.random.default_rng(seed * 100 + 50 + i)
        vis = visible if i == 0 else "bernoulli"
        rbm = RBM(Htr.shape[1], h, vis, rng)
        lr_i = lr * (0.2 if vis == "gauss" else 1.0)
        curves.append(rbm.fit(Htr, Hva, epochs, lr_i, batch, k=k))
        enc.append((rbm.W.copy(), rbm.b.copy()))
        rbms.append(rbm)
        Htr, Hva = rbm.p_h(Htr), rbm.p_h(Hva)
    return enc, curves, rbms


def load_dataset(name, root):
    if name == "ctg":
        cands = [p for p in ("CTG.xlsx", "CTG.xls") if os.path.exists(os.path.join(root, p))]
        if not cands:
            raise SystemExit(f"Не найден CTG.xlsx / CTG.xls в папке {root}")
        X, y = load_ctg(os.path.join(root, cands[0]))
        return dict(name=name, X=X, y=y, K=3, visible="gauss", split="random",
                    classes=["Normal (1)", "Suspect (2)", "Pathologic (3)"],
                    title="Cardiotocography (NSP)")
    if name == "optdigits":
        tr = np.loadtxt(os.path.join(root, "optdigits.tra"), delimiter=",")
        te = np.loadtxt(os.path.join(root, "optdigits.tes"), delimiter=",")
        return dict(name=name, Xtr=tr[:, :-1], ytr=tr[:, -1].astype(int),
                    Xte=te[:, :-1], yte=te[:, -1].astype(int), K=10,
                    visible="bernoulli", split="official",
                    classes=[str(i) for i in range(10)],
                    title="Optical recognition of handwritten digits")
    raise ValueError(name)


def make_split(ds, seed):
    if ds["split"] == "random":
        X, y = ds["X"], ds["y"]
        Xtmp, Xte, ytmp, yte = train_test_split(X, y, test_size=0.20, stratify=y,
                                                random_state=seed)
        Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=0.15,
                                              stratify=ytmp, random_state=seed)
        sc = StandardScaler().fit(Xtr)
        return sc.transform(Xtr), ytr, sc.transform(Xva), yva, sc.transform(Xte), yte
    Xtr, Xva, ytr, yva = train_test_split(ds["Xtr"], ds["ytr"], test_size=0.15,
                                          stratify=ds["ytr"], random_state=seed)
    return Xtr / 16.0, ytr, Xva / 16.0, yva, ds["Xte"] / 16.0, ds["yte"]


def metrics(y, p, K):
    labels = list(range(K))
    return {
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, p, average="weighted", zero_division=0),
        "f1_per_class": f1_score(y, p, labels=labels, average=None, zero_division=0).tolist(),
        "confusion": confusion_matrix(y, p, labels=labels).tolist(),
    }


def run_seed(ds, seed, a):
    Xtr, ytr, Xva, yva, Xte, yte = make_split(ds, seed)
    K = ds["K"]
    sizes = [Xtr.shape[1], *a.hidden, K]

    def make_eval(net):
        pred = net.predict(Xva)
        return {"val_loss": net.loss(Xva, yva),
                "val_f1": f1_score(yva, pred, average="macro", zero_division=0),
                "train_loss": net.loss(Xtr, ytr)}

    def finetune(net):
        return fit(net, Xtr, ytr, a.epochs, a.lr, a.batch, a.l2,
                   np.random.default_rng(seed + 2), make_eval)

    out = {"split": {"train": len(ytr), "val": len(yva), "test": len(yte)}}

    net = MLP(sizes, a.act, "clf", np.random.default_rng(seed + 1))
    h = finetune(net)
    out["scratch"] = {"metrics": metrics(yte, net.predict(Xte), K), "hist": h}

    t0 = time.time()
    enc, ae_curves = pretrain_autoencoders(Xtr, Xva, a.hidden, a.act, a.ae_epochs,
                                           a.ae_lr, a.batch, a.l2, a.noise, seed)
    net = MLP(sizes, a.act, "clf", np.random.default_rng(seed + 1))
    for i, (W, b) in enumerate(enc):
        net.W[i], net.b[i] = W, b
    loss0 = net.loss(Xva, yva)
    t_pre = time.time() - t0
    h = finetune(net)
    out["ae"] = {"metrics": metrics(yte, net.predict(Xte), K), "hist": h,
                 "pre_curves": ae_curves, "val_loss_after_transfer": loss0,
                 "pretrain_seconds": t_pre}

    t0 = time.time()
    enc, rbm_curves, rbms = pretrain_rbms(Xtr, Xva, a.hidden, ds["visible"],
                                          a.rbm_epochs, a.rbm_lr, a.batch, seed,
                                          k=a.rbm_k)
    net = MLP(sizes, a.act, "clf", np.random.default_rng(seed + 1))
    for i, (W, b) in enumerate(enc):
        net.W[i], net.b[i] = W, b
    loss0 = net.loss(Xva, yva)
    t_pre = time.time() - t0
    h = finetune(net)
    out["rbm"] = {"metrics": metrics(yte, net.predict(Xte), K), "hist": h,
                  "pre_curves": rbm_curves, "val_loss_after_transfer": loss0,
                  "pretrain_seconds": t_pre}
    if seed == 0:
        out["rbm"]["W1"] = rbms[0].W.tolist()
    return out


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_plots(res, ds, a, outdir):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "text.color": INK})
    r0, K = res[0], ds["K"]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, ttl in zip(axs, ["train_loss", "val_loss", "val_f1"],
                            ["Loss на обучающей выборке", "Loss на валидации",
                             "Macro-F1 на валидации"]):
        for m in MODES:
            c = np.mean([[h[key] for h in r[m]["hist"]] for r in res], axis=0)
            ax.plot(c, color=COL[m], lw=2, label=LBL[m])
        ax.set_title(ttl, loc="left", fontsize=11)
        ax.set_xlabel("Эпоха дообучения")
        style_ax(ax)
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{ds['title']}: среднее по {len(res)} запускам", x=0.01, ha="left",
                 fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_learning_curves.png"), dpi=150)
    plt.close(fig)

    L = len(a.hidden)
    fig, axs = plt.subplots(2, L, figsize=(3.4 * L, 6.2), squeeze=False)
    for row, (m, nm) in enumerate((("rbm", "RBM"), ("ae", "Автоэнкодер"))):
        for k in range(L):
            ax = axs[row, k]
            tr = np.mean([[h["train"] for h in r[m]["pre_curves"][k]] for r in res], 0)
            va = np.mean([[h["val"] for h in r[m]["pre_curves"][k]] for r in res], 0)
            ax.plot(tr, color=COL[m], lw=2, label="train")
            ax.plot(va, color=COL[m], lw=2, ls="--", label="val")
            ax.set_title(f"{nm}, слой {k + 1} ({a.hidden[k]} нейр.)", loc="left", fontsize=9)
            if row == 1:
                ax.set_xlabel("Эпоха предобучения")
            if k == 0:
                ax.set_ylabel("MSE реконструкции")
                ax.legend(frameon=False, fontsize=8)
            style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_pretraining_reconstruction.png"), dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(5.2 * 3 if K <= 3 else 5.6 * 3, 4.8 if K <= 3 else 5.2))
    fs = 9 if K <= 3 else 6.5
    for ax, m in zip(axs, MODES):
        cm = sum(np.array(r[m]["metrics"]["confusion"]) for r in res)
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(K):
            for j in range(K):
                txt = f"{pct[i, j]:.1f}%\n({cm[i, j]})" if K <= 3 else f"{pct[i, j]:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=fs,
                        color="white" if pct[i, j] > 55 else INK)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(ds["classes"], fontsize=8, rotation=0 if K > 3 else 15)
        ax.set_yticklabels(ds["classes"], fontsize=8)
        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Истинный класс")
        ax.set_title(LBL[m], loc="left", fontsize=11)
    fig.suptitle(f"Confusion matrix (% по строкам), сумма по {len(res)} запускам, тест",
                 x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_confusion_matrices.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    data = [[r[m]["metrics"]["macro_f1"] for r in res] for m in MODES]
    bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False)
    rng = np.random.default_rng(0)
    for k, m in enumerate(MODES):
        bp["boxes"][k].set(facecolor=COL[m], alpha=0.35, edgecolor=COL[m], lw=1.5)
        bp["medians"][k].set(color=COL[m], lw=2)
        ax.scatter(k + 1 + rng.uniform(-0.1, 0.1, len(data[k])), data[k], s=26,
                   color=COL[m], edgecolor="white", linewidth=0.8, zorder=3)
    for w in bp["whiskers"] + bp["caps"]:
        w.set(color=MUTED)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([LBL[m] for m in MODES])
    ax.set_ylabel("Macro-F1 (тест)")
    ax.set_title("Macro-F1 по запускам", loc="left", fontsize=11)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_macro_f1_boxplot.png"), dpi=150)
    plt.close(fig)

    if ds["name"] == "optdigits" and "W1" in r0["rbm"]:
        W = np.array(r0["rbm"]["W1"])
        n = min(W.shape[1], 64)
        g = int(np.ceil(np.sqrt(n)))
        fig, axs = plt.subplots(g, g, figsize=(g * 0.9, g * 0.9))
        for i, ax in enumerate(np.atleast_1d(axs).ravel()):
            ax.axis("off")
            if i < n:
                ax.imshow(W[:, i].reshape(8, 8), cmap="RdBu_r",
                          vmin=-np.abs(W).max(), vmax=np.abs(W).max())
        fig.suptitle("Веса первого слоя RBM (фильтры 8×8)", fontsize=10, x=0.02, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig5_rbm_filters.png"), dpi=150)
        plt.close(fig)


def summarize(res, ds, a, outdir):
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    K = ds["K"]
    L = []
    L.append(f"# {ds['title']}: {len(res)} запусков\n")
    n_in = ds["X"].shape[1] if "X" in ds else ds["Xtr"].shape[1]
    L.append(f"Архитектура: {[n_in, *a.hidden, K]}")
    L.append(f"Активация: {a.act}; дообучение: {a.epochs} эпох, Adam lr={a.lr}, batch={a.batch}, "
             f"L2={a.l2}; AE: {a.ae_epochs} эпох/слой, lr={a.ae_lr}; "
             f"RBM: CD-{a.rbm_k}, {a.rbm_epochs} эпох/слой, lr={a.rbm_lr}, "
             f"видимый слой 1: {ds['visible']}\n")
    L.append("| Метрика | " + " | ".join(LBL[m] for m in MODES) + " |")
    L.append("|---|---|---|---|")
    stats = {}
    for k in keys:
        row = []
        for m in MODES:
            v = np.array([r[m]["metrics"][k] for r in res])
            stats[(m, k)] = v
            row.append(f"{v.mean():.4f} ± {v.std():.4f}")
        L.append(f"| {k} | " + " | ".join(row) + " |")
    L.append("\nF1 по классам:\n")
    L.append("| Класс | " + " | ".join(LBL[m] for m in MODES) + " |")
    L.append("|---|---|---|---|")
    for c, cn in enumerate(ds["classes"]):
        row = []
        for m in MODES:
            v = np.array([r[m]["metrics"]["f1_per_class"][c] for r in res])
            row.append(f"{v.mean():.4f} ± {v.std():.4f}")
        L.append(f"| {cn} | " + " | ".join(row) + " |")

    L.append("\nПопарные сравнения по macro-F1 (критерий Уилкоксона):\n")
    L.append("| Сравнение | Δ среднее | выигрышей / запусков | p |")
    L.append("|---|---|---|---|")
    pairs = [("rbm", "scratch"), ("ae", "scratch"), ("rbm", "ae")]
    for x, y in pairs:
        fx, fy = stats[(x, "macro_f1")], stats[(y, "macro_f1")]
        try:
            from scipy.stats import wilcoxon
            p = f"{wilcoxon(fx, fy).pvalue:.4f}" if np.any(fx != fy) else "н/д"
        except Exception:
            p = "н/д"
        L.append(f"| {LBL[x]} − {LBL[y]} | {np.mean(fx - fy):+.4f} | "
                 f"{int((fx > fy).sum())} / {len(fx)} | {p} |")

    L.append("\nКачество предобучения (MSE реконструкции на val, последняя эпоха, среднее) и "
             "val-loss сети сразу после переноса весов (до дообучения):\n")
    L.append("| Режим | " + " | ".join(f"слой {k + 1}" for k in range(len(a.hidden)))
             + " | val-loss после переноса | время предобучения, с |")
    L.append("|---|" + "---|" * (len(a.hidden) + 2))
    for m in ("ae", "rbm"):
        mse = [np.mean([r[m]["pre_curves"][k][-1]["val"] for r in res]) for k in range(len(a.hidden))]
        l0 = np.mean([r[m]["val_loss_after_transfer"] for r in res])
        t = np.mean([r[m]["pretrain_seconds"] for r in res])
        L.append(f"| {LBL[m]} | " + " | ".join(f"{v:.4f}" for v in mse) + f" | {l0:.4f} | {t:.1f} |")
    text = "\n".join(L)
    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    slim = [{m: r[m]["metrics"] for m in MODES} for r in res]
    with open(os.path.join(outdir, "metrics_per_seed.json"), "w", encoding="utf-8") as fh:
        json.dump(slim, fh, ensure_ascii=False, indent=1)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ctg", "optdigits"], default="ctg")
    ap.add_argument("--root", default=HERE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--act", default="sigmoid", choices=list(ACTS))
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 48, 32, 16])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--ae-epochs", dest="ae_epochs", type=int, default=60)
    ap.add_argument("--ae-lr", dest="ae_lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--rbm-epochs", dest="rbm_epochs", type=int, default=60)
    ap.add_argument("--rbm-lr", dest="rbm_lr", type=float, default=0.05)
    ap.add_argument("--rbm-k", dest="rbm_k", type=int, default=1)
    a = ap.parse_args()

    if a.act != "sigmoid":
        raise SystemExit("Перенос весов RBM в MLP корректен только для --act sigmoid")

    out = a.out or os.path.join(HERE, f"results_{a.dataset}")
    os.makedirs(out, exist_ok=True)
    ds = load_dataset(a.dataset, a.root)
    res = []
    for s in range(a.seeds):
        t0 = time.time()
        r = run_seed(ds, s, a)
        res.append(r)
        print(f"seed {s}: macro-F1  без={r['scratch']['metrics']['macro_f1']:.4f}  "
              f"AE={r['ae']['metrics']['macro_f1']:.4f}  RBM={r['rbm']['metrics']['macro_f1']:.4f}"
              f"  ({time.time() - t0:.0f} c)", flush=True)
    make_plots(res, ds, a, out)
    print("\n" + summarize(res, ds, a, out))


if __name__ == "__main__":
    main()