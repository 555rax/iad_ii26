from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from rbm_lib import (
    DEVICE,
    MLP,
    build_mlp_from_ae,
    build_mlp_from_rbm,
    evaluate_classification,
    make_loader,
    pretrain_ae_stack,
    pretrain_rbm_stack,
    set_seed,
    summarize,
    timestamp,
    train_supervised,
    Logger,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

DATA_PATH = HERE / "agaricus-lepiota.data"
COLUMNS = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat",
]
HIDDEN = (128, 64, 32)
SEEDS = [42, 69, 123, 456, 789]
MODES = ["baseline", "ae", "rbm"]
RBM_EPOCHS = 40
AE_EPOCHS = 100
FT_EPOCHS = 100
PATIENCE = 15


def write_rows(path: Path, rows) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, histories, rbm_histories, confusions_seed42, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    for ax, key, title in zip(
        axes,
        ["accuracy", "f1_macro"],
        ["Accuracy", "F1-macro"],
    ):
        means, stds = [], []
        for mode in MODES:
            vals = [r[key] for r in rows if r["method"] == mode]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        bars = ax.bar(MODES, means, yerr=stds, capsize=5, color=colors)
        ax.set_ylim(0.995, 1.0005)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean,
                    f"{mean:.5f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Mushroom: сравнение способов обучения "
                 f"(среднее ± std, {len(SEEDS)} запусков; ось Y увеличена)")
    fig.tight_layout()
    fig.savefig(out_dir / "mushroom_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, mode in zip(axes, MODES):
        cm = confusions_seed42[mode]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(mode)
        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Истина")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Mushroom: матрицы ошибок (seed=42)")
    fig.tight_layout()
    fig.savefig(out_dir / "mushroom_confusion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in MODES:
        h = histories.get(mode)
        if h is not None:
            ax.plot(h["val_loss"], label=mode)
    ax.set_xlabel("Эпоха дообучения")
    ax.set_ylabel("Validation loss")
    ax.set_title("Mushroom: кривые дообучения (seed=42)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mushroom_finetune_curves.png", dpi=150)
    plt.close(fig)

    if rbm_histories:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for idx, hist in enumerate(rbm_histories):
            ax.plot(hist, label=f"RBM слой {idx + 1}")
        ax.set_xlabel("Эпоха предобучения")
        ax.set_ylabel("Ошибка реконструкции (BCE)")
        ax.set_title("Mushroom: обучение RBM по слоям (seed=42)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "mushroom_rbm_curves.png", dpi=150)
        plt.close(fig)


def main() -> None:
    log = Logger(RESULTS / f"log_mushroom_{timestamp()}.txt")
    log.section("ЛАБОРАТОРНАЯ РАБОТА №4 — MUSHROOM (КЛАССИФИКАЦИЯ)")
    log(f"Устройство: {DEVICE}")
    log(f"Архитектура: in_dim -> {HIDDEN} -> 2")
    log(f"Seed'ы: {SEEDS}")

    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
    y = (df["class"] == "e").astype(int).values.astype(np.int64)
    X_raw = df.drop(columns=["class"])

    log.section("ДАННЫЕ")
    log(f"Исходная таблица: {df.shape}")
    log(f"Баланс классов (edible=1, poisonous=0): {np.bincount(y)}")
    log("Пропуски '?' обрабатываются как отдельная категория (как в ЛР2)")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.2,
        random_state=42, stratify=y_train
    )

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train = encoder.fit_transform(X_train_raw).astype(np.float32)
    X_val = encoder.transform(X_val_raw).astype(np.float32)
    X_test = encoder.transform(X_test_raw).astype(np.float32)
    in_dim = X_train.shape[1]

    log(f"После One-Hot Encoding: {in_dim} признаков")
    log(f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    log(f"Доля единиц в X_train: {X_train.mean():.4f}")

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=DEVICE)

    train_loader = make_loader(X_train, y_train, batch_size=128, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=128, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size=128, shuffle=False)

    rows = []
    histories = {}
    rbm_histories = None
    confusions_seed42 = {}

    for seed in SEEDS:
        for mode in MODES:
            log.section(f"Mushroom | {mode} | seed={seed}")
            set_seed(seed)

            if mode == "baseline":
                set_seed(seed)
                model = MLP(in_dim=in_dim, hidden=HIDDEN, out_dim=2,
                            activation="relu")

            elif mode == "ae":
                aes, _, _, _ = pretrain_ae_stack(
                    X_train_t, X_val_t, HIDDEN,
                    activation="relu", first_sigmoid=True,
                    epochs=AE_EPOCHS, lr=1e-3, batch_size=128,
                    patience=PATIENCE, log_fn=log
                )
                set_seed(seed)
                model = build_mlp_from_ae(in_dim, HIDDEN, 2, aes,
                                          activation="relu")

            else:  # rbm
                rbms, _, _, rb_hist = pretrain_rbm_stack(
                    X_train, X_val, HIDDEN, first_gaussian=False,
                    epochs=RBM_EPOCHS, lr=0.01, batch_size=128,
                    k=1, patience=8, log_fn=log
                )
                if rbm_histories is None:
                    rbm_histories = rb_hist
                set_seed(seed)
                model = build_mlp_from_rbm(in_dim, HIDDEN, 2, rbms,
                                           activation="relu")

            model, hist = train_supervised(
                model, train_loader, val_loader, task="classification",
                epochs=FT_EPOCHS, lr=1e-3, patience=PATIENCE,
                log_fn=log, tag=f"{mode}/seed{seed}"
            )
            metrics = evaluate_classification(model, test_loader)
            log(f"    RESULT {mode} seed={seed}: "
                f"acc={metrics['accuracy']:.4f}, "
                f"f1_macro={metrics['f1_macro']:.4f}, "
                f"f1={metrics['f1']:.4f}")
            log("    CM:\n" + str(metrics["cm"]))

            rows.append({
                "method": mode,
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1": metrics["f1"],
            })
            if seed == SEEDS[0]:
                histories[mode] = hist
                confusions_seed42[mode] = metrics["cm"]

            write_rows(RESULTS / "mushroom_metrics.csv", rows)

    log.section("СВОДКА")
    summary_rows = []
    for mode in MODES:
        subset = [r for r in rows if r["method"] == mode]
        stats = summarize(subset, ["accuracy", "f1_macro", "f1"])
        row = {"method": mode}
        for key, (mean, std) in stats.items():
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
            log(f"{mode:8s} {key:9s}: {mean:.4f} ± {std:.4f}")
        summary_rows.append(row)

    write_rows(RESULTS / "mushroom_summary.csv", summary_rows)
    plot_results(rows, histories, rbm_histories, confusions_seed42, RESULTS)

    log(f"\nМетрики: {RESULTS / 'mushroom_metrics.csv'}")
    log(f"Сводка:  {RESULTS / 'mushroom_summary.csv'}")
    log(f"Графики: {RESULTS}")
    log.close()


if __name__ == "__main__":
    main()
