from __future__ import annotations

from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Logger:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self._f.write(str(msg) + "\n")
        self._f.flush()

    def section(self, title: str) -> None:
        self("\n" + "=" * 68)
        self(f"=== {title} ===")
        self("=" * 68)

    def close(self) -> None:
        self._f.close()

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 64, 32), out_dim: int = 1,
                 activation: str = "leaky_relu"):
        super().__init__()
        act_cls = nn.LeakyReLU if activation == "leaky_relu" else nn.ReLU
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), act_cls()]
            prev = h
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(prev, out_dim)

    def forward(self, x):
        return self.head(self.features(x))


class AE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, use_sigmoid: bool = False,
                 activation: str = "leaky_relu"):
        super().__init__()
        act_cls = nn.LeakyReLU if activation == "leaky_relu" else nn.ReLU
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden_dim), act_cls())
        dec = [nn.Linear(hidden_dim, in_dim)]
        if use_sigmoid:
            dec.append(nn.Sigmoid())
        self.dec = nn.Sequential(*dec)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def feature_linears(model: MLP):
    """Линейные слои внутри model.features (без выходной головы)."""
    return [m for m in model.features if isinstance(m, nn.Linear)]


class RBM(nn.Module):
    def __init__(self, n_visible: int, n_hidden: int,
                 gaussian_visible: bool = False, sigma: float = 1.0):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.gaussian_visible = gaussian_visible
        self.sigma = sigma
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden,
                                          device=DEVICE) * 0.01)
        self.bv = nn.Parameter(torch.zeros(n_visible, device=DEVICE))
        self.bh = nn.Parameter(torch.zeros(n_hidden, device=DEVICE))

    def hidden_prob(self, v: torch.Tensor) -> torch.Tensor:
        if self.gaussian_visible:
            pre = (v / (self.sigma ** 2)) @ self.W + self.bh
        else:
            pre = v @ self.W + self.bh
        return torch.sigmoid(pre)

    def visible_prob(self, h: torch.Tensor) -> torch.Tensor:
        pre = h @ self.W.t() + self.bv
        if self.gaussian_visible:
            return pre
        return torch.sigmoid(pre)

    def sample_h(self, v: torch.Tensor):
        p = self.hidden_prob(v)
        return p, torch.bernoulli(p)

    def sample_v(self, h: torch.Tensor):
        p = self.visible_prob(h)
        if self.gaussian_visible:
            return p, p + self.sigma * torch.randn_like(p)
        return p, torch.bernoulli(p)

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        if self.gaussian_visible:
            vbias = 0.5 * ((v - self.bv) ** 2).sum(dim=1)
            hidden = F.softplus((v / (self.sigma ** 2)) @ self.W + self.bh)
            return vbias - hidden.sum(dim=1)
        vbias = v @ self.bv
        hidden = F.softplus(v @ self.W + self.bh)
        return -vbias - hidden.sum(dim=1)

    def cd_grad(self, v0: torch.Tensor, k: int = 1):
        ph0 = self.hidden_prob(v0)
        hk = torch.bernoulli(ph0)
        vk = v0

        for _ in range(k):
            pvk, vk = self.sample_v(hk)
            phk = self.hidden_prob(vk)
            hk = torch.bernoulli(phk)

        n = v0.size(0)
        dW = (v0.t() @ ph0 - vk.t() @ phk) / n
        dbv = (v0 - vk).mean(dim=0)
        dbh = (ph0 - phk).mean(dim=0)
        return dW, dbv, dbh

    @torch.no_grad()
    def reconstruction_error(self, v: torch.Tensor) -> float:
        ph = self.hidden_prob(v)
        if self.gaussian_visible:
            rec = self.visible_prob(ph)
            return float(F.mse_loss(rec, v).item())
        rec = self.visible_prob(ph).clamp(1e-6, 1 - 1e-6)
        return float(F.binary_cross_entropy(rec, v).item())


def _init_rbm_bias(rbm: RBM, X: torch.Tensor) -> None:
    with torch.no_grad():
        if rbm.gaussian_visible:
            rbm.bv.copy_(X.mean(dim=0))
        else:
            mean = X.mean(dim=0).clamp(1e-3, 1 - 1e-3)
            rbm.bv.copy_(torch.log(mean / (1 - mean)))


def train_rbm(rbm: RBM, X_train: np.ndarray, X_val: np.ndarray,
              epochs: int = 40, lr: float = 0.01, batch_size: int = 128,
              k: int = 1, patience: int = 8, weight_decay: float = 1e-4,
              momentum: float = 0.9, log_fn=None, tag: str = ""):
    Xtr = torch.as_tensor(X_train, dtype=torch.float32, device=DEVICE)
    Xva = torch.as_tensor(X_val, dtype=torch.float32, device=DEVICE)
    _init_rbm_bias(rbm, Xtr)

    vel_W = torch.zeros_like(rbm.W)
    vel_bv = torch.zeros_like(rbm.bv)
    vel_bh = torch.zeros_like(rbm.bh)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    n = Xtr.size(0)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            v0 = Xtr[perm[i:i + batch_size]]
            dW, dbv, dbh = rbm.cd_grad(v0, k=k)

            with torch.no_grad():
                dW = dW - weight_decay * rbm.W
                vel_W.mul_(momentum).add_(dW)
                vel_bv.mul_(momentum).add_(dbv)
                vel_bh.mul_(momentum).add_(dbh)
                rbm.W.add_(vel_W, alpha=lr)
                rbm.bv.add_(vel_bv, alpha=lr)
                rbm.bh.add_(vel_bh, alpha=lr)

        val_err = rbm.reconstruction_error(Xva)
        history.append(val_err)
        if log_fn and (epoch == 1 or epoch % 10 == 0):
            log_fn(f"    {tag} RBM epoch {epoch:3d} | recon {val_err:.5f}")

        if val_err < best_val - 1e-6:
            best_val = val_err
            best_state = {name: p.detach().clone()
                          for name, p in rbm.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if log_fn:
                    log_fn(f"    {tag} RBM early stop at epoch {epoch}")
                break

    if best_state is not None:
        rbm.load_state_dict(best_state)
    return rbm, history


@torch.no_grad()
def rbm_encode(rbm: RBM, X: np.ndarray, sample: bool = False) -> np.ndarray:
    Xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    p = rbm.hidden_prob(Xt)
    if sample:
        p = torch.bernoulli(p)
    return p.cpu().numpy()


def pretrain_rbm_stack(X_train: np.ndarray, X_val: np.ndarray,
                       hidden_dims, first_gaussian: bool,
                       epochs: int = 40, lr: float = 0.01,
                       lr_first: float | None = None,
                       epochs_first: int | None = None,
                       batch_size: int = 128, k: int = 1,
                       patience: int = 8, log_fn=None):

    rbms, histories = [], []
    H_train, H_val = X_train, X_val

    for idx, hidden in enumerate(hidden_dims):
        gaussian = (idx == 0 and first_gaussian)
        rbm = RBM(H_train.shape[1], hidden, gaussian_visible=gaussian)
        layer_lr = lr_first if (idx == 0 and lr_first is not None) else lr
        layer_epochs = (epochs_first if (idx == 0 and epochs_first is not None)
                        else epochs)
        rbm, hist = train_rbm(
            rbm, H_train, H_val, epochs=layer_epochs, lr=layer_lr,
            batch_size=batch_size, k=k, patience=patience,
            log_fn=log_fn, tag=f"layer{idx + 1}"
        )
        rbms.append(rbm)
        histories.append(hist)
        H_train = rbm_encode(rbm, H_train)
        H_val = rbm_encode(rbm, H_val)

    return rbms, H_train, H_val, histories


def build_mlp_from_rbm(in_dim: int, hidden, out_dim: int, rbms,
                       activation: str = "leaky_relu") -> MLP:
    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim,
                activation=activation)
    for linear, rbm in zip(feature_linears(model), rbms):
        with torch.no_grad():
            linear.weight.copy_(rbm.W.t())
            linear.bias.copy_(rbm.bh)
    return model


def train_ae_layer(ae: AE, X_train: torch.Tensor, X_val: torch.Tensor,
                   epochs: int = 100, lr: float = 1e-3,
                   batch_size: int = 128, patience: int = 15,
                   log_fn=None, tag: str = ""):
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size,
                            shuffle=False)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        ae.train()
        for (xb,) in train_loader:
            opt.zero_grad()
            x_hat, _ = ae(xb)
            loss = loss_fn(x_hat, xb)
            loss.backward()
            opt.step()

        ae.eval()
        with torch.no_grad():
            val_loss = float(np.mean([
                loss_fn(ae(xb)[0], xb).item() for (xb,) in val_loader
            ]))
        history.append(val_loss)

        if log_fn and (epoch == 1 or epoch % 20 == 0):
            log_fn(f"    {tag} AE epoch {epoch:3d} | val {val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {name: p.detach().clone()
                          for name, p in ae.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if log_fn:
                    log_fn(f"    {tag} AE early stop at epoch {epoch}")
                break

    if best_state is not None:
        ae.load_state_dict(best_state)
    return ae, history


@torch.no_grad()
def ae_encode(ae: AE, X: torch.Tensor) -> torch.Tensor:
    ae.eval()
    _, z = ae(X)
    return z


def pretrain_ae_stack(X_train: torch.Tensor, X_val: torch.Tensor,
                      hidden_dims, activation: str = "leaky_relu",
                      first_sigmoid: bool = False, epochs: int = 100,
                      lr: float = 1e-3, batch_size: int = 128,
                      patience: int = 15, log_fn=None):
    aes, histories = [], []
    H_train, H_val = X_train, X_val

    for idx, hidden in enumerate(hidden_dims):
        use_sigmoid = (idx == 0 and first_sigmoid)
        ae = AE(H_train.shape[1], hidden, use_sigmoid=use_sigmoid,
                activation=activation).to(DEVICE)
        ae, hist = train_ae_layer(
            ae, H_train, H_val, epochs=epochs, lr=lr,
            batch_size=batch_size, patience=patience,
            log_fn=log_fn, tag=f"layer{idx + 1}"
        )
        aes.append(ae)
        histories.append(hist)
        H_train = ae_encode(ae, H_train)
        H_val = ae_encode(ae, H_val)

    return aes, H_train, H_val, histories


def build_mlp_from_ae(in_dim: int, hidden, out_dim: int, aes,
                      activation: str = "leaky_relu") -> MLP:
    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim,
                activation=activation)
    for linear, ae in zip(feature_linears(model), aes):
        with torch.no_grad():
            linear.weight.copy_(ae.enc[0].weight)
            linear.bias.copy_(ae.enc[0].bias)
    return model

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 64,
                shuffle: bool = True) -> DataLoader:
    Xt = torch.as_tensor(X, dtype=torch.float32)
    if y.dtype.kind == "f":
        yt = torch.as_tensor(y, dtype=torch.float32)
    else:
        yt = torch.as_tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                      shuffle=shuffle)


def _run_epoch(model, loader, opt, loss_fn, task: str):
    model.train()
    total_loss, total_n = 0.0, 0
    total_correct = 0

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()

        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)
        if task == "classification":
            total_correct += (out.argmax(1) == yb).sum().item()

    acc = total_correct / total_n if task == "classification" else None
    return total_loss / total_n, acc


@torch.no_grad()
def _eval_epoch(model, loader, loss_fn, task: str):
    model.eval()
    total_loss, total_n = 0.0, 0
    total_correct = 0

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb)
        loss = loss_fn(out, yb)
        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)
        if task == "classification":
            total_correct += (out.argmax(1) == yb).sum().item()

    acc = total_correct / total_n if task == "classification" else None
    return total_loss / total_n, acc


def train_supervised(model: MLP, train_loader, val_loader, task: str,
                     epochs: int = 150, lr: float = 1e-3,
                     patience: int = 15, log_fn=None, tag: str = ""):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, opt, loss_fn, task)
        va_loss, va_acc = _eval_epoch(model, val_loader, loss_fn, task)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if log_fn and (epoch == 1 or epoch % 10 == 0):
            extra = f" acc {va_acc:.4f}" if va_acc is not None else ""
            log_fn(f"    {tag} epoch {epoch:3d} | train {tr_loss:.4f} "
                   f"| val {va_loss:.4f}{extra}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {name: p.detach().clone()
                          for name, p in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if log_fn:
                    log_fn(f"    {tag} early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_regression(model: MLP, loader, y_scaler):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(DEVICE)).cpu().numpy())
        trues.append(yb.numpy())

    preds = y_scaler.inverse_transform(np.vstack(preds)).ravel()
    trues = y_scaler.inverse_transform(np.vstack(trues)).ravel()

    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    r2 = r2_score(trues, preds)
    mape = np.mean(np.abs((trues - preds) / np.clip(np.abs(trues), 1e-8, None))) * 100
    return {"MAE": float(mae), "RMSE": float(rmse),
            "R2": float(r2), "MAPE": float(mape)}


@torch.no_grad()
def evaluate_classification(model: MLP, loader):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
        trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    return {
        "accuracy": float(accuracy_score(trues, preds)),
        "f1_macro": float(f1_score(trues, preds, average="macro")),
        "f1": float(f1_score(trues, preds)),
        "cm": confusion_matrix(trues, preds),
    }


def summarize(rows, keys):
    return {
        key: (float(np.mean([r[key] for r in rows])),
              float(np.std([r[key] for r in rows])))
        for key in keys
    }


def timestamp() -> str:
    return f"{datetime.now():%Y%m%d_%H%M%S}"
