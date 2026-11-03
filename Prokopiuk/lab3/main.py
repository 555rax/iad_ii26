import pandas as pd
from datetime import datetime
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

LOG_FILE = f"log_{datetime.now():%Y%m%d_%H%M%S}.txt"
RESULTS_CSV = "metrics.csv"

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
    row = {"model": name, **{k: float(v) for k, v in metrics.items()}}
    if extra:
        row.update(extra)
    _results.append(row)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writeheader()
        w.writerows(_results)

class MLP(nn.Module):
    def __init__(self, in_dim=19, h=(128,64,32)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_dim, h[0]),
            nn.LeakyReLU(),
            nn.Linear(h[0], h[1]),
            nn.LeakyReLU(),
            nn.Linear(h[1], h[2]),
            nn.LeakyReLU()
        )
        self.head = nn.Linear(h[2], 1)

    def forward(self, x):
        return self.head(self.features(x))

class AE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.dec = nn.Sequential(nn.Linear(hidden_dim, in_dim))

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat, z


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)

    return total_loss / total_n

@torch.no_grad()
def evaluate_loss(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_n = 0
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)

    return total_loss / total_n

def train_model(model, train_loader, val_loader,
                epochs=100, lr=1e-3, patience=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = evaluate_loss(model, val_loader, loss_fn)

        log(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model

@torch.no_grad()
def evaluate_metrics(model, loader, y_scaler):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb).numpy())
        trues.append(yb.numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)

    preds = y_scaler.inverse_transform(preds).ravel()
    trues = y_scaler.inverse_transform(trues).ravel()

    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    r2 = r2_score(trues, preds)
    mape = np.mean(np.abs((trues - preds) / trues)) * 100

    return {"MAE" : mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def train_ae(ae, X_train_tensor, X_val_tensor,
             epochs=100, lr=1e-3, patience=15):
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    train_ds = TensorDataset(X_train_tensor)
    val_ds =  TensorDataset(X_val_tensor)
    train_loader_ae = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader_ae = DataLoader(val_ds, batch_size=64, shuffle=False)

    for epoch in range(1, epochs+1):
        ae.train()
        for (xb,) in train_loader_ae:
            optimizer.zero_grad()
            x_hat, _ = ae(xb)
            loss = loss_fn(x_hat, xb)
            loss.backward()
            optimizer.step()

        ae.eval()
        with torch.no_grad():
            val_losses = []
            for (xb,) in val_loader_ae:
                x_hat, _ = ae(xb)
                val_losses.append(loss_fn(x_hat, xb).item())
            val_loss = float(np.mean(val_losses))

        log(f"AE epoch {epoch:3d} | val {val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in ae.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"AE early stop at epoch {epoch}")
                break

    ae.load_state_dict(best_state)
    return ae

def build_pretrained_mlp(ae1, ae2, ae3):
    model = MLP(in_dim=19, h=(128, 64, 32))

    # индексы 0, 2, 4 — это nn.Linear внутри features (1, 3, 5 — активации)
    model.features[0].weight.data.copy_(ae1.enc[0].weight.data)
    model.features[0].bias.data.copy_(ae1.enc[0].bias.data)

    model.features[2].weight.data.copy_(ae2.enc[0].weight.data)
    model.features[2].bias.data.copy_(ae2.enc[0].bias.data)

    model.features[4].weight.data.copy_(ae3.enc[0].weight.data)
    model.features[4].bias.data.copy_(ae3.enc[0].bias.data)

    return model

def encode(ae, X_tensor):
    ae.eval()
    with torch.no_grad():
        _, z = ae(X_tensor)
    return z


if __name__ == "__main__":
    data = pd.read_csv("parkinsons_updrs.data")

    y = data["motor_UPDRS"].values
    X = data.drop(columns=["subject#", "motor_UPDRS", "total_UPDRS"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=69
    )

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).ravel()
    y_val = y_scaler.transform(y_val.reshape(-1,1)).ravel()
    y_test = y_scaler.transform(y_test.reshape(-1,1)).ravel()

    X_train = X_train.astype(np.float32)
    X_val   = X_val.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    y_train = y_train.astype(np.float32).reshape(-1, 1)
    y_val = y_val.astype(np.float32).reshape(-1, 1)
    y_test = y_test.astype(np.float32).reshape(-1, 1)

    def make_loader(X, y, batch_size=64, shuffle=True):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    xb, yb = next(iter(train_loader))

    X_train_t = torch.tensor(X_train)
    X_val_t = torch.tensor(X_val)

    SEEDS = [42, 69, 123, 456, 789]

    log_section("MODEL CHECKS")
    model = MLP()
    out = model(xb)
    log(f"MLP out shape: {out.shape}")
    log(f"MLP params: {sum(p.numel() for p in model.parameters())}")

    ae = AE(19, 128)
    x_hat, z = ae(xb)
    log(f"AE x_hat: {x_hat.shape}")
    log(f"AE z:     {z.shape}")
    log(f"AE params: {sum(p.numel() for p in ae.parameters())}")

    def run_experiment(seed, pretrain=True):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if pretrain:
            ae1 = train_ae(AE(19, 128), X_train_t, X_val_t, epochs=100, patience=15)
            H1_tr = encode(ae1, X_train_t)
            H1_va = encode(ae1, X_val_t)

            ae2 = train_ae(AE(128, 64), H1_tr, H1_va, epochs=100, patience=15)
            H2_tr = encode(ae2, H1_tr)
            H2_va = encode(ae2, H1_va)

            ae3 = train_ae(AE(64, 32), H2_tr, H2_va, epochs=100, patience=15)

            model = build_pretrained_mlp(ae1, ae2, ae3)

            if seed == SEEDS[0]:
                log(f"copy check: "
                    f"{torch.equal(model.features[0].weight, ae1.enc[0].weight)}, "
                    f"{torch.equal(model.features[2].weight, ae2.enc[0].weight)}, "
                    f"{torch.equal(model.features[4].weight, ae3.enc[0].weight)}")
        else:
            model = MLP()

        model = train_model(model, train_loader, val_loader, epochs=150, patience=15)
        return evaluate_metrics(model, test_loader, y_scaler)

    log_section("BASELINE")
    for seed in SEEDS:
        log_section(f"baseline, seed={seed}")
        m = run_experiment(seed, pretrain=False)
        log(f"RESULT: {m}")
        record_result(f"baseline_seed{seed}", m, {"seed": seed})

    log_section("PRETRAINED")
    for seed in SEEDS:
        log_section(f"pretrained, seed={seed}")
        m = run_experiment(seed, pretrain=True)
        log(f"RESULT: {m}")
        record_result(f"pretrained_seed{seed}", m, {"seed": seed})

    log_section("SUMMARY")
    def summarize(rows):
        keys = ["MAE", "RMSE", "R2", "MAPE"]
        return {k: (float(np.mean([r[k] for r in rows])),
                    float(np.std([r[k] for r in rows]))) for k in keys}

    base_rows = [r for r in _results if r["model"].startswith("baseline")]
    pre_rows  = [r for r in _results if r["model"].startswith("pretrained")]

    for name, rows in [("BASELINE", base_rows), ("PRETRAINED", pre_rows)]:
        s = summarize(rows)
        log(f"{name} mean±std (n={len(rows)}):")
        for k, (mean, sd) in s.items():
            log(f"  {k:5s}: {mean:.4f} ± {sd:.4f}")

    log(f"\nlog file: {LOG_FILE}")
    log(f"metrics:  {RESULTS_CSV}")
    _log_f.close()