import copy, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

HIDDEN = [64, 32, 16, 8]
ACT = nn.ReLU              
PRETRAIN_EPOCHS = 50       
PRETRAIN_LR = 1e-3
HEAD_EPOCHS = 50           
FINETUNE_EPOCHS = 200      
LR = 1e-3
BATCH = 32
SEEDS = [0, 1, 2, 3, 4]

RBM_LR_GAUSS = 0.01        
RBM_LR_BIN = 0.05          
RBM_WD = 1e-4              
RBM_EPOCHS = 200           
RBM_BATCH = 16

MODES = ["без предобучения", "AE-предобучение", "RBM-предобучение"]


def T(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)

def make_mlp(in_dim, hidden, out_dim):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), ACT()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers).to(device)

def batches(n, batch):
    idx = torch.randperm(n, device=device)
    for i in range(0, n, batch):
        yield idx[i:i + batch]

def train_supervised(model, Xtr, ytr, Xva, yva, loss_fn, epochs, lr, params=None):
    opt = torch.optim.Adam(model.parameters() if params is None else params, lr=lr)
    best, best_state = float("inf"), None
    hist = {"train": [], "val": []}
    for ep in range(epochs):
        total = 0.0
        for idx in batches(len(Xtr), BATCH):
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        with torch.no_grad():
            val = loss_fn(model(Xva), yva).item()
        hist["train"].append(total / len(Xtr))
        hist["val"].append(val)
        if val < best:
            best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return hist

@torch.no_grad()
def predict(model, X):
    return model(X).squeeze(1).cpu().numpy()

def pretrain_ae(model, Xtr, Xva):
    linears = [m for m in model if isinstance(m, nn.Linear)]
    h_tr, h_va = Xtr, Xva
    for enc in linears[:-1]:
        in_d, out_d = enc.in_features, enc.out_features
        ae = nn.Sequential(enc, ACT(), nn.Linear(out_d, in_d)).to(device)
        train_supervised(ae, h_tr, h_tr, h_va, h_va, nn.MSELoss(),
                         PRETRAIN_EPOCHS, PRETRAIN_LR)
        with torch.no_grad():
            h_tr, h_va = ACT()(enc(h_tr)), ACT()(enc(h_va))

def train_head(model, Xtr, ytr, Xva, yva, loss_fn):
    last = [m for m in model if isinstance(m, nn.Linear)][-1]
    train_supervised(model, Xtr, ytr, Xva, yva, loss_fn,
                     HEAD_EPOCHS, LR, params=last.parameters())

def reg_metrics(y_log_true, y_log_pred):
    a_true = np.expm1(y_log_true)
    a_pred = np.clip(np.expm1(y_log_pred), 0, None)
    m = a_true > 0
    return {"MAE": mean_absolute_error(a_true, a_pred),
            "MedAE": float(np.median(np.abs(a_true - a_pred))),   
            "RMSE": float(np.sqrt(mean_squared_error(a_true, a_pred))),
            "MAPE_%": float(np.mean(np.abs((a_true[m] - a_pred[m]) / a_true[m])) * 100),
            "R2_log": r2_score(y_log_true, y_log_pred)}

def evaluate(task, y_true, out):
    if task == "reg":
        return reg_metrics(y_true, out), None
    pred = (out > 0).astype(int)
    return ({"Accuracy": accuracy_score(y_true, pred),
             "F1_macro": f1_score(y_true, pred, average="macro")},
            confusion_matrix(y_true, pred))


class RBM:
    def __init__(self, n_vis, n_hid, gaussian_visible):
        self.W = 0.01 * torch.randn(n_vis, n_hid, device=device)  
        self.a = torch.zeros(n_vis, device=device)                
        self.b = torch.zeros(n_hid, device=device)                
        self.gauss = gaussian_visible   

    def hidden_prob(self, v):           
        return torch.sigmoid(v @ self.W + self.b)

    def visible_mean(self, h):          
        z = h @ self.W.T + self.a
        return z if self.gauss else torch.sigmoid(z)

    @torch.no_grad()
    def cd1(self, v0, lr):
        ph0 = self.hidden_prob(v0)              
        h0 = torch.bernoulli(ph0)               
        v1 = self.visible_mean(h0)              
        ph1 = self.hidden_prob(v1)              
        n = v0.shape[0]                         
        self.W += lr * ((v0.T @ ph0 - v1.T @ ph1) / n - RBM_WD * self.W)
        self.a += lr * (v0 - v1).mean(0)
        self.b += lr * (ph0 - ph1).mean(0)

    @torch.no_grad()
    def recon_error(self, v):
        return ((v - self.visible_mean(self.hidden_prob(v))) ** 2).mean().item()

    def fit(self, Xtr, Xva, epochs, lr, batch):
        best, best_state, hist = float("inf"), None, []
        for ep in range(epochs):
            for idx in batches(len(Xtr), batch):
                self.cd1(Xtr[idx], lr)
            e = self.recon_error(Xva)
            hist.append(e)
            if e < best:                         
                best = e
                best_state = (self.W.clone(), self.a.clone(), self.b.clone())
        self.W, self.a, self.b = best_state
        return hist

def pretrain_rbm(model, Xtr, Xva, log=None):
    linears = [m for m in model if isinstance(m, nn.Linear)]
    v_tr, v_va = Xtr, Xva
    for k, lin in enumerate(linears[:-1]):          
        rbm = RBM(lin.in_features, lin.out_features, gaussian_visible=(k == 0))
        lr = RBM_LR_GAUSS if k == 0 else RBM_LR_BIN
        hist = rbm.fit(v_tr, v_va, RBM_EPOCHS, lr, RBM_BATCH)
        with torch.no_grad():
            lin.weight.copy_(rbm.W.T)               
            lin.bias.copy_(rbm.b)
            v_tr, v_va = rbm.hidden_prob(v_tr), rbm.hidden_prob(v_va)
        if log is not None:
            log.append(hist)


def run_experiment(title, X, y, task, class_names=None):
    loss_fn = nn.MSELoss() if task == "reg" else nn.BCEWithLogitsLoss()
    rows, curves, cms, rbm_curves = [], {}, {}, []

    for seed in SEEDS:
        strat = y if task == "clf" else None
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=strat)
        X_va, X_te, y_va, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=seed,
            stratify=y_tmp if task == "clf" else None)

        sc = StandardScaler().fit(X_tr)
        Xtr, Xva, Xte = [T(sc.transform(a)) for a in (X_tr, X_va, X_te)]
        ytr, yva = T(y_tr).unsqueeze(1), T(y_va).unsqueeze(1)

        for mode in MODES:
            torch.manual_seed(seed)                 
            model = make_mlp(Xtr.shape[1], HIDDEN, 1)
            rbm_log = []
            if mode == "AE-предобучение":
                pretrain_ae(model, Xtr, Xva)
                train_head(model, Xtr, ytr, Xva, yva, loss_fn)
            elif mode == "RBM-предобучение":
                pretrain_rbm(model, Xtr, Xva, log=rbm_log)
                train_head(model, Xtr, ytr, Xva, yva, loss_fn)
            hist = train_supervised(model, Xtr, ytr, Xva, yva, loss_fn, FINETUNE_EPOCHS, LR)
            metrics, cm = evaluate(task, y_te, predict(model, Xte))
            rows.append({"режим": mode, "seed": seed, **metrics})
            if seed == SEEDS[0]:
                curves[mode], cms[mode] = hist["val"], cm
                if mode == "RBM-предобучение":
                    rbm_curves = rbm_log
        print(f"seed {seed} готов")

    df = pd.DataFrame(rows)
    print(f"\n=== {title}: среднее и std по {len(SEEDS)} запускам, тестовая выборка ===")
    display(df.drop(columns="seed").groupby("режим", sort=False).agg(["mean", "std"]).round(3))

    plt.figure(figsize=(6, 4))
    for k, h in enumerate(rbm_curves, 1):
        plt.plot(h, label=f"RBM {k}")
    plt.yscale("log"); plt.xlabel("эпоха RBM"); plt.ylabel("ошибка реконструкции (val)")
    plt.title(f"{title}: обучение слоёв как RBM"); plt.legend(); plt.grid(alpha=.3); plt.show()

    plt.figure(figsize=(6, 4))
    for mode, v in curves.items():
        plt.plot(v, label=mode)
    plt.xlabel("эпоха основного обучения"); plt.ylabel("loss на val")
    plt.title(title); plt.legend(); plt.grid(alpha=.3); plt.show()

    if task == "clf":
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, (mode, cm) in zip(axs, cms.items()):
            ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False)
            ax.set_title(mode)
        plt.tight_layout(); plt.show()
    return df


RBM_EPOCHS, RBM_BATCH = 200, 16      

ff = pd.read_csv(r"D:\forest+fires\forestfires.csv")
print(ff.shape)
area = ff["area"]
feat = ff.drop(columns="area")
X_ff = pd.get_dummies(feat, columns=["month", "day"], dtype=float).to_numpy(dtype=np.float32)
y_ff = np.log1p(area.to_numpy())

results = {}
for ACT in (nn.ReLU, nn.Sigmoid):
    results[("Forest Fires", ACT.__name__)] = run_experiment(
        f"Forest Fires, {ACT.__name__}", X_ff, y_ff, task="reg")


RBM_EPOCHS, RBM_BATCH = 100, 64      

RICE_FILE = r"D:\rice+cammeo+and+osmancik\Rice_Cammeo_Osmancik.arff"
ext = os.path.splitext(RICE_FILE)[1].lower()
if ext in (".xlsx", ".xls"):
    rice = pd.read_excel(RICE_FILE)
elif ext == ".arff":
    from scipy.io import arff
    data, meta = arff.loadarff(RICE_FILE)
    rice = pd.DataFrame(data)
else:
    rice = pd.read_csv(RICE_FILE, sep=None, engine="python")

feat = rice.iloc[:, :-1]
target = rice.iloc[:, -1]
if isinstance(target.iloc[0], bytes):
    target = target.str.decode("utf-8")
target = target.astype(str).str.strip()
classes = sorted(target.unique())
print(rice.shape, classes, target.value_counts().to_dict())

X_rice = feat.to_numpy(dtype=np.float32)
y_rice = (target == classes[1]).astype(int).to_numpy()

for ACT in (nn.ReLU, nn.Sigmoid):
    results[("Rice", ACT.__name__)] = run_experiment(
        f"Rice, {ACT.__name__}", X_rice, y_rice, task="clf", class_names=classes)


frames = []
for (name, act), df in results.items():
    t = df.drop(columns="seed").groupby("режим", sort=False).mean().round(3)
    t.insert(0, "активация", act)
    t.insert(0, "датасет", name)
    frames.append(t.reset_index())
display(pd.concat(frames, ignore_index=True))