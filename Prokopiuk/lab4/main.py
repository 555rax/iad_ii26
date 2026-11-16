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
from sklearn.preprocessing import StandardScaler

from rbm_lib import (
    DEVICE,
    MLP,
    build_mlp_from_ae,
    build_mlp_from_rbm,
    evaluate_regression,
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

DATA_PATH = HERE / "parkinsons_updrs.data"
HIDDEN = (128, 64, 32)
SEEDS = [42, 69, 123, 456, 789]
MODES = ["baseline", "ae", "rbm"]
RBM_EPOCHS = 40
RBM_LR_FIRST = 0.001
RBM_EPOCHS_FIRST = 60
AE_EPOCHS = 100
FT_EPOCHS = 150
PATIENCE = 15


def write_rows(path: Path, rows) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, histories, rbm_histories, out_dir: Path) -> None:
    keys = ["MAE", "RMSE", "R2"]
    labels = {"MAE": "MAE (меньше — лучше)",
              "RMSE": "RMSE (меньше — лучше)",
              "R2": "R² (больше — лучше)"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    for ax, key in zip(axes, keys):
        means, stds = [], []
        for mode in MODES:
            vals = [r[key] for r in rows if r["method"] == mode]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.bar(MODES, means, yerr=stds, capsize=5, color=colors)
        ax.set_title(labels[key])
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Parkinsons: сравнение способов обучения "
                 f"(среднее ± std, {len(SEEDS)} запусков)")
    fig.tight_layout()
    fig.savefig(out_dir / "parkinsons_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    data = [[r["R2"] for r in rows if r["method"] == m] for m in MODES]
    ax.boxplot(data, labels=MODES)
    ax.set_ylabel("R²")
    ax.set_title("Parkinsons: разброс R² по запускам")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "parkinsons_r2_boxplot.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in MODES:
        h = histories.get(mode)
        if h is not None:
            ax.plot(h["val_loss"], label=mode)
    ax.set_xlabel("Эпоха дообучения")
    ax.set_ylabel("Validation MSE")
    ax.set_title("Parkinsons: кривые дообучения (seed=42)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "parkinsons_finetune_curves.png", dpi=150)
    plt.close(fig)

    if rbm_histories:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for idx, hist in enumerate(rbm_histories):
            ax.plot(hist, label=f"RBM слой {idx + 1}")
        ax.set_xlabel("Эпоха предобучения")
        ax.set_ylabel("Ошибка реконструкции")
        ax.set_title("Parkinsons: обучение RBM по слоям (seed=42)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "parkinsons_rbm_curves.png", dpi=150)
        plt.close(fig)


def main() -> None:
    log = Logger(RESULTS / f"log_parkinsons_{timestamp()}.txt")
    log.section("ЛАБОРАТОРНАЯ РАБОТА №4 — PARKINSONS (РЕГРЕССИЯ)")
    log(f"Устройство: {DEVICE}")
    log(f"Архитектура: {19} -> {HIDDEN} -> 1")
    log(f"Seed'ы: {SEEDS}")

    data = pd.read_csv(DATA_PATH)
    y = data["motor_UPDRS"].values.astype(np.float32)
    X = data.drop(columns=["subject#", "motor_UPDRS", "total_UPDRS"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=69
    )

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train).astype(np.float32)
    X_val = x_scaler.transform(X_val).astype(np.float32)
    X_test = x_scaler.transform(X_test).astype(np.float32)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_val = y_scaler.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel().astype(np.float32)

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    log.section("ДАННЫЕ")
    log(f"X: {X.shape}, y: {y.shape}")
    log(f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    log(f"X_train: mean={X_train.mean():.4f}, std={X_train.std():.4f}")

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=DEVICE)

    train_loader = make_loader(X_train, y_train, batch_size=64, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=64, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size=64, shuffle=False)

    rows = []
    histories = {}
    rbm_histories = None

    for seed in SEEDS:
        for mode in MODES:
            log.section(f"Parkinsons | {mode} | seed={seed}")
            set_seed(seed)

            if mode == "baseline":
                set_seed(seed)
                model = MLP(in_dim=19, hidden=HIDDEN, out_dim=1,
                            activation="leaky_relu")

            elif mode == "ae":
                aes, _, _, _ = pretrain_ae_stack(
                    X_train_t, X_val_t, HIDDEN,
                    activation="leaky_relu", first_sigmoid=False,
                    epochs=AE_EPOCHS, lr=1e-3, batch_size=128,
                    patience=PATIENCE, log_fn=log
                )
                set_seed(seed)
                model = build_mlp_from_ae(19, HIDDEN, 1, aes,
                                          activation="leaky_relu")

            else:
                rbms, _, _, rb_hist = pretrain_rbm_stack(
                    X_train, X_val, HIDDEN, first_gaussian=True,
                    epochs=RBM_EPOCHS, lr=0.01,
                    lr_first=RBM_LR_FIRST,
                    epochs_first=RBM_EPOCHS_FIRST,
                    batch_size=128, k=1, patience=8, log_fn=log
                )
                if rbm_histories is None:
                    rbm_histories = rb_hist
                set_seed(seed)
                model = build_mlp_from_rbm(19, HIDDEN, 1, rbms,
                                           activation="leaky_relu")

            model, hist = train_supervised(
                model, train_loader, val_loader, task="regression",
                epochs=FT_EPOCHS, lr=1e-3, patience=PATIENCE,
                log_fn=log, tag=f"{mode}/seed{seed}"
            )
            metrics = evaluate_regression(model, test_loader, y_scaler)
            log(f"    RESULT {mode} seed={seed}: "
                f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
                f"R2={metrics['R2']:.4f}, MAPE={metrics['MAPE']:.2f}%")

            rows.append({"method": mode, "seed": seed, **metrics})
            if seed == SEEDS[0]:
                histories[mode] = hist

            write_rows(RESULTS / "parkinsons_metrics.csv", rows)

    log.section("СВОДКА")
    summary_rows = []
    for mode in MODES:
        subset = [r for r in rows if r["method"] == mode]
        stats = summarize(subset, ["MAE", "RMSE", "R2", "MAPE"])
        row = {"method": mode}
        for key, (mean, std) in stats.items():
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
            log(f"{mode:8s} {key:5s}: {mean:.4f} ± {std:.4f}")
        summary_rows.append(row)

    write_rows(RESULTS / "parkinsons_summary.csv", summary_rows)
    plot_results(rows, histories, rbm_histories, RESULTS)

    log(f"\nМетрики: {RESULTS / 'parkinsons_metrics.csv'}")
    log(f"Сводка:  {RESULTS / 'parkinsons_summary.csv'}")
    log(f"Графики: {RESULTS}")
    log.close()


if __name__ == "__main__":
    main()
