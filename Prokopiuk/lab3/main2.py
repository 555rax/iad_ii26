import pandas as pd
from datetime import datetime
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

LOG_FILE = f"log_mushroom_{datetime.now():%Y%m%d_%H%M%S}.txt"
RESULTS_CSV = "metrics_mushroom.csv"

_log_f = open(LOG_FILE, "w", encoding="utf-8")
_results = []

def log(msg):
    print(msg)
    _log_f.write(str(msg) + "\n")
    _log_f.flush()

def log_section(title):
    log("\n" + "=" * 60)
    log(f"=== {title} ===")
    log("=" * 60)

def record_result(name, metrics, extra=None):
    row = {"model": name, **{k: v for k, v in metrics.items() if k != "cm"}}
    if extra:
        row.update(extra)
    _results.append(row)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writeheader()
        w.writerows(_results)


class MLP(nn.Module):
    def __init__(self, in_dim, h=(128, 64, 32), n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_dim, h[0]), nn.ReLU(),
            nn.Linear(h[0], h[1]),   nn.ReLU(),
            nn.Linear(h[1], h[2]),   nn.ReLU(),
        )
        self.head = nn.Linear(h[2], n_classes)

    def forward(self, x):
        return self.head(self.features(x))


class AE(nn.Module):
    def __init__(self, in_dim, hidden_dim, use_sigmoid=False):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )
        layers = [nn.Linear(hidden_dim, in_dim)]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.dec = nn.Sequential(*layers)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return total / n, correct / n


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn):
    model.eval()
    total, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return total / n, correct / n


def train_model(model, train_loader, val_loader,
                epochs=100, lr=1e-3, patience=15):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn)
        val_loss, val_acc = evaluate_loss(model, val_loader, loss_fn)

        log(f"epoch {epoch:3d} | train {tr_loss:.4f} acc {tr_acc:.4f} "
            f"| val {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log(f"early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_metrics(model, loader):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb).argmax(1).numpy())
        trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    return {
        "acc": float(accuracy_score(trues, preds)),
        "f1":  float(f1_score(trues, preds)),
        "cm":  confusion_matrix(trues, preds),
    }


def train_ae(ae, X_train_t, X_val_t, epochs=100, lr=1e-3, patience=15):
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tl = DataLoader(TensorDataset(X_train_t), batch_size=128, shuffle=True)
    vl = DataLoader(TensorDataset(X_val_t),   batch_size=128, shuffle=False)

    best_val, best_state, bad = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        ae.train()
        for (xb,) in tl:
            opt.zero_grad()
            x_hat, _ = ae(xb)
            loss = loss_fn(x_hat, xb)
            loss.backward()
            opt.step()

        ae.eval()
        with torch.no_grad():
            vlosses = [loss_fn(ae(xb)[0], xb).item() for (xb,) in vl]
            val_loss = float(np.mean(vlosses))

        log(f"AE epoch {epoch:3d} | val {val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in ae.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log(f"AE early stop at epoch {epoch}")
                break

    ae.load_state_dict(best_state)
    return ae


def build_pretrained_mlp(ae1, ae2, ae3, in_dim):
    model = MLP(in_dim=in_dim, h=(128, 64, 32))
    model.features[0].weight.data.copy_(ae1.enc[0].weight.data)
    model.features[0].bias.data.copy_(ae1.enc[0].bias.data)
    model.features[2].weight.data.copy_(ae2.enc[0].weight.data)
    model.features[2].bias.data.copy_(ae2.enc[0].bias.data)
    model.features[4].weight.data.copy_(ae3.enc[0].weight.data)
    model.features[4].bias.data.copy_(ae3.enc[0].bias.data)
    return model


def encode(ae, X_t):
    ae.eval()
    with torch.no_grad():
        _, z = ae(X_t)
    return z


if __name__ == "__main__":
    cols = ['class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
            'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color',
            'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
            'stalk_surface_below_ring', 'stalk_color_above_ring',
            'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number',
            'ring_type', 'spore_print_color', 'population', 'habitat']

    df = pd.read_csv("agaricus-lepiota.data", header=None, names=cols)
    df = df.replace('?', np.nan).dropna()

    y = df['class'].map({'p': 1, 'e': 0}).values.astype(np.int64)
    X = pd.get_dummies(df.drop(columns=['class'])).values.astype(np.float32)
    in_dim = X.shape[1]

    log_section("DATA")
    log(f"X: {X.shape}, y balance: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    log(f"split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    def make_loader(X_, y_, bs=128, shuffle=True):
        return DataLoader(TensorDataset(torch.tensor(X_), torch.tensor(y_)),
                          batch_size=bs, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    X_train_t = torch.tensor(X_train)
    X_val_t   = torch.tensor(X_val)

    SEEDS = [42, 69, 123, 456, 789]

    def run_experiment(seed, pretrain):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if pretrain:
            ae1 = train_ae(AE(in_dim, 128, True), X_train_t, X_val_t)
            H1_tr = encode(ae1, X_train_t); H1_va = encode(ae1, X_val_t)

            ae2 = train_ae(AE(128, 64), H1_tr, H1_va)
            H2_tr = encode(ae2, H1_tr); H2_va = encode(ae2, H1_va)

            ae3 = train_ae(AE(64, 32), H2_tr, H2_va)

            model = build_pretrained_mlp(ae1, ae2, ae3, in_dim)
            if seed == SEEDS[0]:
                log(f"copy check: "
                    f"{torch.equal(model.features[0].weight, ae1.enc[0].weight)}, "
                    f"{torch.equal(model.features[2].weight, ae2.enc[0].weight)}, "
                    f"{torch.equal(model.features[4].weight, ae3.enc[0].weight)}")
        else:
            model = MLP(in_dim=in_dim, h=(128, 64, 32))

        model = train_model(model, train_loader, val_loader,
                            epochs=100, patience=15)
        return evaluate_metrics(model, test_loader)

    log_section("BASELINE")
    for seed in SEEDS:
        log_section(f"baseline, seed={seed}")
        m = run_experiment(seed, pretrain=False)
        log(f"RESULT: acc={m['acc']:.4f}, f1={m['f1']:.4f}")
        log(f"CM:\n{m['cm']}")
        record_result(f"baseline_seed{seed}", m, {"seed": seed})

    log_section("PRETRAINED")
    for seed in SEEDS:
        log_section(f"pretrained, seed={seed}")
        m = run_experiment(seed, pretrain=True)
        log(f"RESULT: acc={m['acc']:.4f}, f1={m['f1']:.4f}")
        log(f"CM:\n{m['cm']}")
        record_result(f"pretrained_seed{seed}", m, {"seed": seed})

    log_section("SUMMARY")
    def summarize(rows):
        return {k: (float(np.mean([r[k] for r in rows])),
                    float(np.std([r[k] for r in rows]))) for k in ["acc", "f1"]}

    base = [r for r in _results if r["model"].startswith("baseline")]
    pre  = [r for r in _results if r["model"].startswith("pretrained")]

    for name, rows in [("BASELINE", base), ("PRETRAINED", pre)]:
        s = summarize(rows)
        log(f"{name} mean±std (n={len(rows)}):")
        for k, (m, sd) in s.items():
            log(f"  {k}: {m:.4f} ± {sd:.4f}")

    log(f"\nlog: {LOG_FILE}")
    log(f"metrics: {RESULTS_CSV}")
    _log_f.close()