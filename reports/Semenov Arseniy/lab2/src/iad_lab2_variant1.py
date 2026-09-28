

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from torch import nn

SEED = 42
PERPLEXITIES = (20, 40, 60)
CONFIGS = [((16,), .001), ((16,), .003),
           ((32, 16), .001), ((32, 16), .003)]


def prepare(x):
    """Параметры стандартизации вычисляются только по переданной выборке."""
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=1)
    if not np.isfinite(x).all() or np.any(std <= 0):
        raise ValueError('Нужны конечные значения и непостоянные признаки.')
    return (x - mean) / std, mean, std


def manual_pca(z, k):
    """PCA через numpy.linalg.eig, как в лабораторной работе 1."""
    mean = z.mean(axis=0)
    centered = z - mean
    cov = centered.T @ centered / (len(z) - 1)
    values, vectors = np.linalg.eig(cov)
    values, vectors = np.real_if_close(values), np.real_if_close(vectors)
    if np.iscomplexobj(values) or np.iscomplexobj(vectors):
        raise ArithmeticError('Комплексный результат разложения ковариации.')
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if values.min() < -1e-9:
        raise ArithmeticError('Отрицательное собственное значение ковариации.')
    values = np.maximum(values, 0)
    basis = vectors[:, :k]
    signs = np.sign(basis[np.argmax(abs(basis), axis=0), np.arange(k)])
    basis = basis * signs
    scores = centered @ basis
    if not np.allclose(basis.T @ basis, np.eye(k), atol=1e-9):
        raise ArithmeticError('Базис PCA не ортонормирован.')
    return scores, basis, values, mean


class Autoencoder(nn.Module):
    """Полносвязная сеть с линейным узким слоем и линейным выходом."""

    def __init__(self, input_dim, k, hidden):
        super().__init__()
        enc, dec = [], []
        widths = [input_dim, *hidden, k]
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            enc.append(nn.Linear(a, b))
            if i < len(widths) - 2:
                enc.append(nn.ReLU())
        widths = [k, *reversed(hidden), input_dim]
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            dec.append(nn.Linear(a, b))
            if i < len(widths) - 2:
                dec.append(nn.ReLU())
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_ae(train, k, hidden, lr, epochs, seed, valid=None, patience=150):
    torch.manual_seed(seed)
    model = Autoencoder(train.shape[1], k, hidden).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.from_numpy(np.ascontiguousarray(train, dtype=np.float64))
    v = None if valid is None else torch.from_numpy(
        np.ascontiguousarray(valid, dtype=np.float64))
    history, best, best_epoch, state = [], float('inf'), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = (model(x) - x).square().mean()
        if not torch.isfinite(loss):
            raise ArithmeticError('Обучение расходится: MSE не конечна.')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float((model(x) - x).square().mean())
            val_loss = None if v is None else float((model(v) - v).square().mean())
        history.append([epoch, train_loss, val_loss])
        if v is not None:
            if val_loss < best:
                best, best_epoch = val_loss, epoch
                state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= patience:
                break
    if v is not None:
        model.load_state_dict(state)
    else:
        best_epoch = epochs
    return model, history, best_epoch


def encode_reconstruct(model, z):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float64))
        encoded = model.encoder(x).numpy()
        reconstructed = model(x).numpy()
    if not np.isfinite(encoded).all() or not np.isfinite(reconstructed).all():
        raise ArithmeticError('Неконечные координаты автоэнкодера.')
    return encoded, reconstructed


def mse(x, reconstructed):
    return float(np.mean((x - reconstructed) ** 2))


def split_indices(n):
    train, holdout = train_test_split(np.arange(n), test_size=.4, random_state=SEED)
    valid, test = train_test_split(holdout, test_size=.5, random_state=SEED)
    return train, valid, test


def experiments(out, epochs):
    dataset = load_breast_cancer()
    x, y = dataset.data.astype(np.float64), dataset.target
    assert x.shape == (569, 30) and np.isfinite(x).all()
    assert np.array_equal(np.bincount(y), [212, 357])
    # ID отсутствует в load_breast_cancer; diagnosis хранится отдельно в target.
    # Метки не передаются ни в обучение, ни в подбор гиперпараметров.
    train, valid, test = split_indices(len(x))
    ztrain, train_mean, train_std = prepare(x[train])
    zvalid, ztest = (x[valid] - train_mean) / train_std, (x[test] - train_mean) / train_std
    z, full_mean, full_std = prepare(x)
    report = dict(seed=SEED, epochs_limit=epochs, rows=len(x), features=x.shape[1],
                  class_counts={'M': 212, 'B': 357},
                  split_sizes=[len(train), len(valid), len(test)],
                  versions={'numpy': np.__version__, 'sklearn': sklearn.__version__,
                            'torch': torch.__version__}, candidates=[], pca={}, ae={}, tsne=[])
    projections, histories = {}, {}
    for k in (2, 3):
        print(f'\nАвтоэнкодер: {k} координаты', flush=True)
        candidates = []
        for hidden, lr in CONFIGS:
            model, hist, best_epoch = train_ae(
                ztrain, k, hidden, lr, epochs, SEED, valid=zvalid)
            row = dict(k=k, hidden=list(hidden), lr=lr, best_epoch=best_epoch,
                       epochs_run=len(hist), train_mse=mse(ztrain, encode_reconstruct(model, ztrain)[1]),
                       valid_mse=mse(zvalid, encode_reconstruct(model, zvalid)[1]))
            report['candidates'].append(row)
            candidates.append((row, model, hist))
            print(row, flush=True)
        selected, selected_model, histories[k] = min(candidates, key=lambda c: c[0]['valid_mse'])
        # Тестовая часть впервые используется после выбора по validation.
        test_mse = mse(ztest, encode_reconstruct(selected_model, ztest)[1])
        # Для итоговой визуализации переобучаем выбранную архитектуру на всех
        # объектах. Число эпох взято из validation; test его не определяет.
        final_model, _, _ = train_ae(z, k, selected['hidden'], selected['lr'],
                                    selected['best_epoch'], SEED)
        encoded, reconstruction = encode_reconstruct(final_model, z)
        projections[('ae', k)] = encoded
        report['ae'][str(k)] = dict(**selected, test_mse=test_mse,
            full_mse=mse(z, reconstruction),
            trustworthiness=float(trustworthiness(z, encoded, n_neighbors=10)))
        torch.save({'state_dict': final_model.state_dict(), 'hidden': selected['hidden'],
                    'k': k, 'mean': full_mean.tolist(), 'std': full_std.tolist()},
                   out / f'autoencoder_{k}d.pt')

        scores, basis, eigenvalues, center = manual_pca(z, k)
        library = PCA(n_components=k, svd_solver='full').fit(z)
        pca_reconstructed = scores @ basis.T + center
        theoretical = (len(z)-1) * eigenvalues[k:].sum() / z.size
        projector_error = float(np.max(abs(basis @ basis.T -
                                    library.components_.T @ library.components_)))
        assert projector_error < 1e-8
        assert np.allclose(eigenvalues[:k], library.explained_variance_)
        assert np.isclose(mse(z, pca_reconstructed), theoretical)
        _, train_basis, _, train_center = manual_pca(ztrain, k)
        test_reconstructed = (ztest - train_center) @ train_basis @ train_basis.T + train_center
        projections[('pca', k)] = scores
        report['pca'][str(k)] = dict(
            retained_percent=float(100 * eigenvalues[:k].sum() / eigenvalues.sum()),
            full_mse=mse(z, pca_reconstructed), test_mse=mse(ztest, test_reconstructed),
            trustworthiness=float(trustworthiness(z, scores, n_neighbors=10)),
            projector_error=projector_error, eigenvalues=eigenvalues.tolist())

        for perplexity in PERPLEXITIES:
            print(f't-SNE: k={k}, perplexity={perplexity}', flush=True)
            estimator = TSNE(n_components=k, perplexity=perplexity, init='pca',
                             learning_rate='auto', max_iter=1500, random_state=SEED,
                             method='barnes_hut', n_jobs=1)
            embedding = estimator.fit_transform(z)
            assert embedding.shape == (len(x), k) and np.isfinite(embedding).all()
            row = dict(k=k, perplexity=perplexity, kl=float(estimator.kl_divergence_),
                       iterations=int(estimator.n_iter_),
                       trustworthiness=float(trustworthiness(z, embedding, n_neighbors=10)))
            report['tsne'].append(row)
            projections[('tsne', k, perplexity)] = embedding
        best = max((r for r in report['tsne'] if r['k'] == k),
                   key=lambda r: r['trustworthiness'])
        report['ae'][str(k)]['selected_tsne_perplexity'] = best['perplexity']

    for key, values in projections.items():
        np.savetxt(out / ('_'.join(map(str, key)) + '.csv'), np.column_stack([values, y]),
                   delimiter=',', header=','.join([f'coordinate_{j+1}' for j in range(values.shape[1])] + ['class_0_M_1_B']), comments='')
    for k, hist in histories.items():
        with (out / f'learning_{k}d.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_mse', 'validation_mse'])
            writer.writerows(hist)
    (out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report, projections, histories, y


def plot_results(report, projections, histories, y, out, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

    def scatter(ax, points, title, prefix):
        for cls, color, marker, label in [(0, '#be423d', '^', 'M — злокачественные'),
                                           (1, '#2869a6', 'o', 'B — доброкачественные')]:
            a = points[y == cls]
            ax.scatter(*[a[:, j] for j in range(a.shape[1])], c=color, marker=marker,
                       s=15, alpha=.72, label=label)
        ax.set(xlabel=prefix+'1', ylabel=prefix+'2', title=title)
        if points.shape[1] == 3:
            from matplotlib.ticker import MaxNLocator
            ax.set_zlabel(prefix+'3')
            ax.view_init(elev=22, azim=40)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(nbins=4))
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=.2)

    for method, name, prefix in [('ae', 'Автоэнкодер', 'h'), ('pca', 'PCA', 'PC')]:
        fig = plt.figure(figsize=(10, 4.5), layout='constrained')
        for panel, k in enumerate((2, 3), 1):
            ax = fig.add_subplot(1, 2, panel, projection='3d' if k == 3 else None)
            scatter(ax, projections[(method, k)], f'{name}, {k} координаты', prefix)
        fig.savefig(out / f'{method}.png', dpi=190)

    for k in (2, 3):
        fig = plt.figure(figsize=(10, 8), layout='constrained')
        for panel, p in enumerate(PERPLEXITIES, 1):
            ax = fig.add_subplot(2, 2, panel, projection='3d' if k == 3 else None)
            score = next(r['trustworthiness'] for r in report['tsne'] if r['k'] == k and r['perplexity'] == p)
            scatter(ax, projections[('tsne', k, p)], f'p = {p}; T(10) = {score:.4f}', 't')
        fig.suptitle(f't-SNE, {k} координаты; влияние perplexity')
        fig.savefig(out / f'tsne_{k}d.png', dpi=190)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, k in zip(axes, (2, 3)):
        h = np.array(histories[k])
        ax.plot(h[:, 0], h[:, 1], label='Обучение')
        ax.plot(h[:, 0], h[:, 2], label='Валидация')
        ax.axvline(report['ae'][str(k)]['best_epoch'], ls='--', color='gray', label='Выбранная эпоха')
        ax.set(xlabel='Эпоха', ylabel='MSE', title=f'Автоэнкодер, k = {k}', yscale='log')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(out / 'learning.png', dpi=190)
    if show:
        plt.show()
    plt.close('all')


def self_test():
    x, _ = load_breast_cancer(return_X_y=True)
    a, b, c = split_indices(len(x))
    assert not (set(a) & set(b) or set(a) & set(c) or set(b) & set(c))
    assert len(set(a) | set(b) | set(c)) == len(x)
    z, _, _ = prepare(x)
    assert np.allclose(z.mean(0), 0, atol=1e-12)
    assert np.allclose(z.std(0, ddof=1), 1)
    for k in (2, 3, 30):
        scores, basis, values, mean = manual_pca(z, k)
        pca = PCA(k, svd_solver='full').fit(z)
        assert np.allclose(basis @ basis.T, pca.components_.T @ pca.components_, atol=1e-8)
        assert np.isclose(mse(z, scores @ basis.T + mean), (len(z)-1)*values[k:].sum()/z.size, atol=1e-10)
    model, hist, _ = train_ae(z[:100], 2, (16,), .003, 100, SEED)
    assert hist[-1][1] < hist[0][1]
    emb, rec = encode_reconstruct(model, z[:100])
    assert emb.shape == (100, 2) and rec.shape == (100, 30)
    model2, _, _ = train_ae(z[:100], 2, (16,), .003, 100, SEED)
    assert np.array_equal(emb, encode_reconstruct(model2, z[:100])[0])
    print('Проверки пройдены: PCA, реконструкция, стандартизация, разбиение, обучение, воспроизводимость.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'results_lab2')
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs должен быть положительным')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        if args.self_test:
            self_test()
            return
        args.out.mkdir(parents=True, exist_ok=True)
        report, projections, histories, y = experiments(args.out, args.epochs)
        plot_results(report, projections, histories, y, args.out, not args.no_show)
    for k in ('2', '3'):
        print(f'k={k}: PCA сохраняет {report["pca"][k]["retained_percent"]:.4f}% дисперсии; '
              f'MSE test: PCA={report["pca"][k]["test_mse"]:.6f}, AE={report["ae"][k]["test_mse"]:.6f}')
    print(f'Результаты сохранены: {args.out.resolve()}')


if __name__ == '__main__':
    main()
