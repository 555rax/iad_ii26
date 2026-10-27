from pathlib import Path
import argparse
import copy
import hashlib
import io
import json
import platform
import time
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import torch
from torch import nn
import lab3_base as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

class Network(base.Network):
    def __init__(self, dimensions, activation='sigmoid'):
        super().__init__(dimensions)
        self.activation = activation
        if activation == 'sigmoid':
            for layer in self.hidden:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.hidden:
            x = self.activate(layer(x))
        return self.head(x)

    def activate(self, x):
        return torch.sigmoid(x) if self.activation == 'sigmoid' else torch.relu(x)


def download_data():
    folder = ROOT / 'data'
    folder.mkdir(exist_ok=True)
    sources = [
        ('energydata_complete.csv', 'https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip'),
        ('optdigits.tra', 'https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip')]
    for name, url in sources:
        if not (folder / name).exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                archive = zipfile.ZipFile(io.BytesIO(response.read()))
            (folder / name).write_bytes(archive.read(name))


def pretrain_ae(model, x, seed, epochs):
    h, history = x.clone(), []
    for number, layer in enumerate(model.hidden, 1):
        base.seed_all(seed + 1000 + number)
        encoder = copy.deepcopy(layer)
        activation = nn.Sigmoid() if model.activation == 'sigmoid' else nn.ReLU()
        ae = nn.Sequential(encoder, activation, nn.Linear(layer.out_features, layer.in_features))
        optimizer = torch.optim.Adam(ae.parameters(), lr=.001)
        generator = torch.Generator().manual_seed(seed + number)
        for epoch in range(1, epochs + 1):
            total = 0.
            for ix in base.batches(len(h), generator):
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(ae(h[ix]), h[ix])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(ix)
            history.append(dict(layer=number, epoch=epoch, mse=total / len(h)))
        layer.load_state_dict(encoder.state_dict())
        with torch.no_grad():
            h = model.activate(layer(h)).detach()
    return history


@torch.no_grad()
def pretrain_rbm(model, x, seed, epochs, gaussian_first=False):
    h, history = x.clone(), []
    for number, layer in enumerate(model.hidden, 1):
        base.seed_all(seed + 2000 + number)
        gaussian = gaussian_first and number == 1
        w = torch.randn_like(layer.weight) * .01
        vb = torch.zeros(layer.in_features)
        hb = torch.zeros(layer.out_features)
        vw, vv, vh = torch.zeros_like(w), torch.zeros_like(vb), torch.zeros_like(hb)
        generator = torch.Generator().manual_seed(seed + number)
        lr = .001 if gaussian else .05
        for epoch in range(1, epochs + 1):
            total = 0.
            momentum = .5 if epoch <= 5 else .9
            for ix in base.batches(len(h), generator):
                values = h[ix]
                v0 = values if gaussian else torch.bernoulli(values.clamp(0, 1))
                ph0 = torch.sigmoid(v0 @ w.T + hb)
                h0 = torch.bernoulli(ph0)
                linear = h0 @ w + vb
                v1 = linear + torch.randn_like(linear) if gaussian else torch.bernoulli(torch.sigmoid(linear))
                ph1 = torch.sigmoid(v1 @ w.T + hb)
                vw = momentum * vw + lr * ((ph0.T @ v0 - ph1.T @ v1) / len(ix) - .0001 * w)
                vv = momentum * vv + lr * (v0 - v1).mean(0)
                vh = momentum * vh + lr * (ph0 - ph1).mean(0)
                w += vw
                vb += vv
                hb += vh
                recon_linear = ph0 @ w + vb
                recon = recon_linear if gaussian else torch.sigmoid(recon_linear)
                total += (recon - values).square().mean().item() * len(ix)
            if not torch.isfinite(w).all():
                raise RuntimeError('RBM diverged')
            history.append(dict(layer=number, epoch=epoch, mse=total / len(h)))
        layer.weight.copy_(w)
        layer.bias.copy_(hb)
        torch.save(dict(weight=w, visible_bias=vb, hidden_bias=hb, gaussian=gaussian),
                   OUT / f'rbm_{len(x)}_{seed}_layer{number}.pt')
        h = torch.sigmoid(layer(h)).detach()
    return history


def run(data, seeds, epochs):
    rows = []
    xs = [torch.from_numpy(x) for x in data['xs']]
    ys = [torch.from_numpy(y) for y in data['ys']]
    classification = data['task'] == 'classification'
    for seed in seeds:
        for activation in ['relu', 'sigmoid']:
            base.seed_all(seed)
            initial = Network(data['dimensions'], activation)
            modes = ['baseline', 'ae'] + (['rbm'] if activation == 'sigmoid' else [])
            for mode in modes:
                model = copy.deepcopy(initial)
                prefix = f"{data['name']}_{seed}_{activation}_{mode}"
                started = time.perf_counter()
                if mode != 'baseline':
                    pre = (pretrain_ae(model, xs[0], seed, epochs) if mode == 'ae' else
                           pretrain_rbm(model, xs[0], seed, epochs, not classification))
                    pd.DataFrame(pre).to_csv(OUT / f'{prefix}_pretrain.csv', index=False)
                    for param in model.hidden.parameters():
                        param.requires_grad_(False)
                    warm, _ = base.supervised(model, xs[0], ys[0], xs[1], ys[1], seed, classification, True)
                    pd.DataFrame(warm).to_csv(OUT / f'{prefix}_warm.csv', index=False)
                    for param in model.hidden.parameters():
                        param.requires_grad_(True)
                history, best = base.supervised(model, xs[0], ys[0], xs[1], ys[1], seed, classification)
                seconds = time.perf_counter() - started
                with torch.no_grad():
                    raw = model(xs[2]).numpy()
                pred = raw.argmax(1) if classification else data['sy'].inverse_transform(raw).ravel()
                row = dict(dataset=data['name'], seed=seed, activation=activation, mode=mode,
                           **base.evaluate(data['task'], data['truth'], pred), best_epoch=best,
                           epochs_run=len(history), seconds=seconds)
                rows.append(row)
                print(json.dumps(row), flush=True)
                pd.DataFrame(history).to_csv(OUT / f'{prefix}_history.csv', index=False)
                pd.DataFrame(dict(source_row=data['indices'][2], actual=data['truth'], predicted=pred)).to_csv(
                    OUT / f'{prefix}_predictions.csv', index=False)
                torch.save(dict(state_dict=model.state_dict(), dimensions=data['dimensions'],
                                activation=activation, metadata=data['metadata']), OUT / f'{prefix}.pt')
                pd.DataFrame(rows).to_csv(OUT / f"{data['name']}_metrics.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--pretrain-epochs', type=int, default=50)
    args = parser.parse_args()
    download_data()
    datasets = [base.prepare_energy(), base.prepare_digits()]
    metadata = dict(python=platform.python_version(), torch=torch.__version__, seeds=args.seeds,
                    pretrain_epochs=args.pretrain_epochs,
                    datasets={d['name']: d['metadata'] for d in datasets},
                    sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').iterdir() if p.is_file()})
    (OUT/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    for data in datasets:
        rows.extend(run(data, args.seeds, args.pretrain_epochs))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'metrics_all.csv', index=False)
    metrics = [c for c in frame.select_dtypes('number').columns if c != 'seed']
    frame.groupby(['dataset', 'activation', 'mode'])[metrics].agg(['mean', 'std']).to_csv(OUT/'metrics_summary.csv')
    from report import make_report
    make_report()


if __name__ == '__main__':
    main()
