

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
import zlib

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Сжатая копия seeds_dataset.txt из предоставленного архива.
# Проверка SHA-256 исключает случайное изменение встроенных данных.
DATA_SHA256 = '1f3f83c0d8485ae9148061389d19628607e3f5660e3d6f40ec9102fb398bb12f'
DATA_B85 = (
    'c-mE(Te4d_2t>bq2Xjh7@A*&cs#mkUou5=H#xjDYX;A9;i}|VjZ}(66+pB)Y-<s{y{#uPs{6*+rB0lv$8Eu)4'
    'CDUE?$rxR>t$pfWsr%Fa&X_iiBja+e@hL;6oQmzb=gxDk?T#VareD8*9(*`M#w|E_9XT;DE*Q_h+I8^p!L488'
    'IrcRk^Sb%h_<CK#UNl_3x8P*C-1huluivlyxIA&WA3|||;NFi|$bozQ{o=hvJvP4g*Pk3rEPCkNqI1{hGd}kA'
    '9#HRRlQaAJ%)c^pW}o947P-IIvTg6Oy+KiYw?5g5!qCnC&cq>w`oH>38@BVvvsGmrT~`{t>8c8qd@T)M?F7cX'
    'DP&RZCGHp3{g<M@lME*=zFzdE>(`?{&kaKVTMDae7vU{Fc*w!NFdhnWZ8F{~(wM}R;ZW%2eJf`5x*30`#S4jh'
    '1H6b$d=Ix7HL-TDaxZ?}*y-|hO^!D~qHY=2DmNN`j|-Bo@yIN%VPNWlBbLIfXx-n{ElN*FtS@{{9UON3SN5g?'
    'H`hIQb`Wv19`eHFF%6bb7)lp@YXNIdN-FsS@ngRrt|7n6JxU;+zfcQDIRcMmJGV-S93?DA)+i7@d@?J=QRLQ|'
    'lt4Mv9+xeTqv7%Fe00c#uGg!Ad}hY2;Kjm=IRu%t4~P>y*QqavZjl~A*@)X+A>9qwJC=onmgV#n0+%Ouiu5ST'
    'h48i2USI?$YJZa`4}6w!P{ncYsa)^*Z;^tcP2jpfBwD*C4Q~8kS=V@QKPk*nN#B$&qd9WvveI`EXXB}p`vAq@'
    'Y7fWR#4<!FN{Tr)bo3gFtI3;Q1FY;UR<jVyGU8C#GOiHU!S<X4J|$I`4+hJ7P=D8#$}g1~nLq9~iN#tz7s`r_'
    'vHqpaOyB*^Ww4v6kL-%_+_w%5CBFA&yMVC#^HA|7Wqm3ZJ6aakwOxQs^h^GiA)()&w@95n5ki)SJB>MrV<wCe'
    'n<61#tz?p8zd<<Nh)}@d>`S+Ek6(5hz{kAle%Ja=#=7v?i^5Atj33((Uk<865es0QMfxCOm{tlOeK}U0routV'
    '-lSsBFD%spZYB=o>dtC7jD+xTZ-i9_6sz7PzEIL@`fD~e%npHA@bFNDj`>^7ZNAmKCw;%buSdPSP2$k9@gE>L'
    'Yz(I+d>5J9za4j-pVU=)Rv%=8nxn94t(j)La52xN%gsg5#t$Y(8u#|4%td_}>#-CF72Z9#cVEwq>9{=`Kuhhh'
    'gVI4msV(`{<vCq(ZBrRkQKi?^38d1vwYM>CNI{(Q^v6a%A`-?o3_97A0A<2II{5PIMmFq~^~u4@u^^_a=w_i}'
    'yp-Is729P3*9^_N(KY*<l3ju(mR=mZ=*`gNHVBK($+cs=6K_^bc4Oo71ZiEAevxpvUzm)Q$NLe+4Qi@rr6T0q'
    'z@SCknyLiS{fbdPK~X76GrV#o(+!k6LRCw;rn9+y2)F2}Z9O>EPgiO7p;37Cys0kBZI|c;=cdpMI*rBd+;5{%'
    'AFo_C;}9U|>E;Uv1aVMPs+7IzIpeTxY*KOUBZI2^Z6n^h-K4?UEjb&^H}Ipns{k`g$sISi*$m-(mh<vY{ijA7'
    'Ql*{u*fDJk*4!ySdACNx#orFpI(m4c#|BU<GkMzz_H5T^`>45B^tLEXv7S<Zp?<JYuvyRK;+#$IP!cCKDy|D1'
    'tUwO=B9AA=6#qCf9*q_y*7&a;kT$J>PPv71y<qk@E%nNELg>H?+R~t&*XAQSXh*yji|$@+Eb)C^b|<I!4{}fn'
    'xx3vi2@HAMIM}Qrp{R>fNJSak^xJbW7XQT+Ti4^2;!4+>N!+{YdVS~)`tzQ=Y2@=6J%&DkG$%pOkIaknuou<W'
    'Gum(v9vpL0OfW~!gbb3``rAl#yt>C?n&8{pK1jdLr;rw-N>L%11zLDkIldjX``8D$7dWxOSv5@da_;W;73lsb'
    'kD2_l^oy50LSMgMu`!8WnLO@|6SUDYgM)`X4h}B%LJsSPZ`1}Yw;QX=wkpvC(AF$<zh@_=WWPlGH7+O9f$=nP'
    '>yy!yx+!4HL*|a67M0$+o(Xefi(oBO5jTid9kh*Yu|{YH0H!c-6;fNIEnzqYMAs){tLoNf7#zk(x+@JLF@_7B'
    'IH-^{>PaHx(CIL^j-2E&i99OoFzL4SYpfw`KK@P$sV%Y$IO<!?8ycS3_xsRqP7>D{)zrS^@XiW#(Rq69ijfJk'
    'LxrAa&rJ9<TprGH)x{fKlz4Q28g2h3?wP?>21&Y1{3dyznCJECdLc2nc_7G#$#Aq6mrr$MB`I~mh&i{UZr!>='
    'G1jKU`#Z=_59csjC>ADe&1IDHVN`NkJT7MrK03J2rFP3ubO+gqRwSe(G_tG1w1N}H@Bx@Jd#EM|RgyhM=MC-~'
    '58wy<lY(0)4rK8fs_HgLQO_Vt#eOlw$%civ_G;+SzC<BoPDY&{4MZaDprD~EiZnhBg<DPPVvg1A21TvqiP$x7'
    'XTm&T!US#*+sciN{$(Uh3|t6bZW1<jFp`9oWrkwjZQSjziz#R{h09H}M;kPRp3o%5D;|+>V@xYpbVzAg;do?1'
    '2wTem7&o?zt!*kZ!-+h(azATQ&HOXBPz#*S{TB7*3QeW!!JGMtFq~6qyClC(pIpFDPh$##Id>se!qqqv_2<hh'
    'cV;s;P<ls-sWp$X>GAkE*9f8`Af=0G=^^e7u|z6TI5M6fKV9#rpXqjmxSeO~DDG|9mNcC=STnnlQBm!KudSay'
    'jOeG3&Zplw#Lo0u4k%n<5~}ownlhoT@AP_GEIk%E*tQ}O3}%v~#*ih`dcuZ$Qoo~zvF_1MG^DpZTWzm1t<w1Y'
    'ffj3R=2&AAGB|T9M$4+?14&%*OB{mnbV90Eg!|j0t*_cKwo_CbrgP7_-nX4)wpgrSxkSz8uUU5~FU<Wy@_>hy'
    'gyo+P-l>^Xd?Q=wVM&z5;e7-8ccuz*`4fs-_SXlIP(Xr%O#UdQ<~r@jg5FVKY=iuDH^^tGaWiEA@@6%rG+v)S'
    'B5!iJp)rr%WUUAZ9?08telmMwezTYQYB7BDf_h|+xBL3_;t2)I(v_>=8;?)UcUGF##$NfE>2YE(#SZlSSv8#i'
    'o7J1AFiq0T1-JRiHs#^tl|o6O=oL&r+=~%IWJ(-@Y~=3D!kszL(qf*k)~yUswIv&O=7}XvJ6=MRXG_;)+u~jJ'
    'Pv#vj<;n$*Ks8t}rL7+ZtP@zkZ0|=Yp)+!e*fi95wlg5+h#bQM9ia`p2iGd&%81asO9&2YETPv@HFx$x+5%x4'
    '47n9RR1OQ~Q0t58gV(TeZ?AD{BR53xuq*E=n%Qu<v$yW^9r<Ifbh)!D)r_3RgxI>cpXVjx+Cl&TLKJ^Z3p*H1'
    'MR;6L@0Dj4=Y~ku(=!z9T>|&xLATMGvHB;*g-}@JdH>Z5(>qh3)k{42U@o5y1j`Fg`z-Jm>%KPaLS=mR;>rak'
    'V5KH)R>vhX=W*Y!2yI8Mo&u84@@wY>eDHaV!~1*L<{%3AkZ~N`Sx<PeOdRO13W~hwek5tWT=fJ^(8~788UP{a'
    'H<c9OgV%V2R`9rs&yfe3+u1|p?TiRZm6$E7N1+(G<9Tc&SD7BRJHN;{NwKTr@=GU;y<<144X@uJRSAUf9<=N<'
    '#b5W0o|_N0N?oY*tAZX#?8JE2Bm)R5*@7_CTAv-M(?N*Wd5!6>=&6~v(`$)COd<KT$oo2=b~L!NRrP}1g+wSk'
    'A{gF{db3VBjIIwZtFPzf2k&H^;@D_duK-%h+HV+JBgCoSa-LkTwe^;BMSZjnz~n7VN<3GI!?nEgSh9g--Sk0V'
    'xzH!etnonD0d_L{wwcLLQ&@hY85v*A!(rXA9eVv@Ac{K+^YKZXQ5SG~^X9b8G}K;|MC@ZW*(~D@m0Z04Tf98+'
    'R8w7pd!(^!6AlNs0Lftey}Ve;o0%X=b--}KeF!uXg>oVp&B!sWvKCf9ln=o!khitx;J)9g@BC&Z3hsEUnZK+W'
    'KO4+GTo=TbJ@}mebOOzFk?jZqbB33bu(I-v$oUqLB}X*2lGQ~7aQXvDs=j&M!2ZyktMKrR_R`J|A0~Jxj<8er'
    'Y*F<(>E&ZOu{qbf+(EYb_hHQv?b|eXACMDxT+dD+8obZwjW;scZW{G1A{+PG$Q@6<{`$z|9m1Aii!UlCajWb6'
    '17X1ChqZxx5Dtx?D<-37s5yAh=ThPJ%`cM?MET+T<vv%`4!>SJZ@iI@-|d)s#%~>Y4jq(4;q_v(=|A5WC5M8>'
    '?03U;MiPQxf{Z^u6j9I};g{+QmH9xE`rN=T%kJ5&`*F<^QBb44R?VGJ!g#+ZW)!L3{P_L7<-@aO1eh4@S&mvJ'
    'sa*UbkFmklSSs_%|9!@7&#M#9HVzEu&iLs@e%YvBdD5~1(smhY9%KCO%#2?x^f`TJr)|EP<s$G7sQv7dI`f-2'
    'LF(0jXSvdNoT1q@y)mpO$rY;d?9P^j&51jrrreGruk<p^Hfqs>yY8-q!Ax3>j91UcX{2JY1{H41$ff=ky<y}3'
    'v%r6iIjm=9!-4frQF!>|dC+Q=iKNANm5B~i8h)PY%sX*|vYCdzZL{Y8yNP^iu3s&+`O3(mNZ_D&T0Wi7P&<vD'
    'fJ{0e)c!wDz$l3'
)


def read_data(path=None):
    if path is None:
        raw = zlib.decompress(base64.b85decode(DATA_B85))
        if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
            raise ValueError('Нарушена целостность встроенных данных.')
    elif path.suffix.lower() == '.zip':
        with ZipFile(path) as archive:
            names = [n for n in archive.namelist()
                     if Path(n).name == 'seeds_dataset.txt']
            if len(names) != 1:
                raise ValueError('В архиве нужен один seeds_dataset.txt.')
            raw = archive.read(names[0])
    else:
        raw = path.read_bytes()
    frame = pd.read_csv(io.BytesIO(raw), sep=r'\s+', header=None,
                        na_values=['?', 'NA', 'NaN', 'nan'],
                        on_bad_lines='error')
    if frame.shape[1] != 8 or len(frame) < 3:
        raise ValueError('Ожидаются >=3 строк и 8 колонок: 7 признаков и класс.')
    frame = frame.apply(pd.to_numeric, errors='raise')
    labels = frame.iloc[:, -1].to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isin(labels, [1, 2, 3]).all():
        raise ValueError('Классы должны быть 1, 2 или 3 без пропусков.')
    # Класс удаляется ДО любых вычислений PCA.
    x = frame.iloc[:, :-1].to_numpy(dtype=float)
    if np.isinf(x).any():
        raise ValueError('Бесконечные значения признаков недопустимы.')
    missing = np.isnan(x)
    if np.any(missing.all(axis=0)):
        raise ValueError('Нельзя заполнить полностью пустой признак.')
    medians = np.nanmedian(x, axis=0)
    x = np.where(missing, medians, x)
    return x, labels.astype(int), {
        'sha256': hashlib.sha256(raw).hexdigest(),
        'missing_by_feature': missing.sum(axis=0).tolist(),
        'rows': len(x), 'features': x.shape[1],
        'class_counts': {str(int(c)): int(np.sum(labels == c))
                         for c in np.unique(labels)},
    }


def preprocess(x, scale='standard'):
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    if np.any(std == 0):
        raise ValueError('PCA этой работы требует непостоянные признаки.')
    divisor = std if scale == 'standard' else np.ones(x.shape[1])
    return (x - mean) / divisor, mean, divisor


def manual_pca(z):
    # Именно eig, как требуется в методичке; меток классов здесь нет.
    covariance = z.T @ z / (len(z) - 1)
    values, vectors = np.linalg.eig(covariance)
    values = np.real_if_close(values, tol=1000)
    vectors = np.real_if_close(vectors, tol=1000)
    if np.iscomplexobj(values) or np.iscomplexobj(vectors):
        raise ArithmeticError('Значимая мнимая часть при разложении ковариации.')
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if values.min() < -1e-10:
        raise ArithmeticError('Ковариация имеет отрицательное собственное значение.')
    values = np.maximum(values, 0.)
    if not np.allclose(vectors.T @ vectors, np.eye(z.shape[1]), atol=1e-9):
        raise ArithmeticError('Базис eig не ортонормирован.')
    return values, vectors, covariance


def canonical_signs(components, scores):
    # Знак собственного вектора произволен. Для сопоставимых рисунков
    # наибольшая по модулю координата каждого вектора делается положительной.
    rows = np.argmax(np.abs(components), axis=0)
    signs = np.sign(components[rows, np.arange(components.shape[1])])
    return components * signs, scores * signs


def analyze(z):
    values, vectors, covariance = manual_pca(z)
    total = float(values.sum())
    projections, report = {}, {}
    for k in (2, 3):
        basis, scores = canonical_signs(vectors[:, :k], z @ vectors[:, :k])
        reconstructed = scores @ basis.T
        # Независимое решение: PCA получает исходную подготовленную матрицу.
        # Результаты eig не передаются библиотечной реализации.
        estimator = PCA(n_components=k, svd_solver='full', whiten=False)
        library_scores = estimator.fit_transform(z)
        library_basis, library_scores = canonical_signs(
            estimator.components_.T, library_scores)
        library_reconstructed = library_scores @ library_basis.T + estimator.mean_
        discarded = float(values[k:].sum())
        sse = float(np.sum((z - reconstructed) ** 2))
        library_total = float(np.var(z, axis=0, ddof=1).sum())
        report[str(k)] = {
            'retained_percent': float(100 * values[:k].sum() / total),
            'lost_percent': 100 * discarded / total,
            'discarded_variance': discarded,
            'reconstruction_sse': sse,
            'reconstruction_mse': sse / z.size,
            'theoretical_sse': (len(z) - 1) * discarded,
            'sklearn_retained_percent': float(
                100 * estimator.explained_variance_ratio_.sum()),
            'sklearn_lost_percent': float(100 * (
                1 - estimator.explained_variance_.sum() / library_total)),
            'sklearn_sse': float(np.sum((z - library_reconstructed) ** 2)),
            'eigenvalues_max_error': float(np.max(np.abs(
                values[:k] - estimator.explained_variance_))),
            'projector_max_error': float(np.max(np.abs(
                basis @ basis.T - library_basis @ library_basis.T))),
            'scores_max_error': float(np.max(np.abs(scores - library_scores))),
            'basis': basis.tolist(),
        }
        projections[k] = (scores, library_scores)
    return {'eigenvalues': values.tolist(),
            'ratios_percent': (100 * values / total).tolist(),
            'covariance': covariance.tolist(), 'dimensions': report}, projections


def visualize(results, projections, labels, out=None, show=True):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    styles = [(1, '#2264aa', 'o'), (2, '#dc751a', '^'), (3, '#39854a', 's')]
    for k in (2, 3):
        fig = plt.figure(figsize=(11, 4.7), layout='constrained')
        data_pair = projections[k]
        mins = np.vstack(data_pair).min(axis=0)
        maxs = np.vstack(data_pair).max(axis=0)
        span = np.maximum(maxs - mins, 1e-9)
        for panel, (name, scores) in enumerate(zip(
                ('NumPy eig', 'sklearn PCA'), data_pair), 1):
            ax = fig.add_subplot(1, 2, panel, projection='3d' if k == 3 else None)
            for cls, color, marker in styles:
                points = scores[labels == cls]
                coords = [points[:, j] for j in range(k)]
                ax.scatter(*coords, color=color, marker=marker, s=23,
                           alpha=.8, label=f'Класс {cls}')
            ax.set(xlabel='PC1', ylabel='PC2', title=name,
                   xlim=(mins[0]-.06*span[0], maxs[0]+.06*span[0]),
                   ylim=(mins[1]-.06*span[1], maxs[1]+.06*span[1]))
            if k == 3:
                ax.set_zlabel('PC3')
                ax.set_zlim(mins[2]-.06*span[2], maxs[2]+.06*span[2])
                ax.view_init(elev=22, azim=40)
            else:
                ax.grid(alpha=.2)
            ax.legend(fontsize=9, loc='upper left' if k == 3 else 'lower left')
        keep = results['dimensions'][str(k)]['retained_percent']
        fig.suptitle(f'Первые {k} главные компоненты; сохранено {keep:.2f}% дисперсии')
        if out:
            fig.savefig(out / f'pca_{k}d.png', dpi=200, bbox_inches='tight')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    positions = np.arange(1, 8)
    axes[0].bar(positions, results['eigenvalues'], color='#2264aa')
    axes[0].set(xlabel='Номер компоненты', ylabel='Собственное значение',
                title='Спектр ковариационной матрицы', xticks=positions)
    cumulative = np.cumsum(results['ratios_percent'])
    axes[1].plot(positions, cumulative, 'o-', label='Сохранённая дисперсия')
    axes[1].axhline(95, color='#dc751a', ls='--', label='Уровень 95%')
    axes[1].set(xlabel='Число компонент', ylabel='Доля дисперсии, %',
                title='Накопленная объяснённая дисперсия', xticks=positions,
                ylim=(0, 105))
    axes[1].grid(alpha=.2); axes[1].legend(fontsize=9)
    if out:
        fig.savefig(out / 'spectrum.png', dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close('all')


def self_test():
    import tempfile
    x, labels, info = read_data()
    assert x.shape == (210, 7)
    assert np.array_equal(np.unique(labels, return_counts=True)[1], [70]*3)
    assert sum(info['missing_by_feature']) == 0
    for scale in ('standard', 'center'):
        z, _, _ = preprocess(x, scale)
        report, _ = analyze(z)
        assert np.allclose(z.mean(axis=0), 0, atol=1e-12)
        if scale == 'standard':
            assert np.allclose(z.std(axis=0, ddof=1), 1)
            assert np.isclose(sum(report['eigenvalues']), 7)
        values, basis, cov = manual_pca(z)
        assert np.allclose(cov @ basis, basis * values)
        assert np.allclose(z @ basis @ basis.T, z)
        for r in report['dimensions'].values():
            assert r['projector_max_error'] < 1e-9
            assert r['scores_max_error'] < 1e-9
            assert r['eigenvalues_max_error'] < 1e-9
            assert np.isclose(r['reconstruction_sse'], r['theoretical_sse'])
            assert np.isclose(r['reconstruction_sse'], r['sklearn_sse'])
    # Проверка загрузки ZIP/TXT и заполнения явного пропуска медианой.
    raw = zlib.decompress(base64.b85decode(DATA_B85))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'seeds_dataset.txt'; path.write_bytes(raw)
        assert np.array_equal(read_data(path)[0], x)
        archive = Path(tmp) / 'seeds.zip'
        with ZipFile(archive, 'w') as f:
            f.writestr('seeds_dataset.txt', raw)
        assert np.array_equal(read_data(archive)[0], x)
        data = np.c_[x, labels]; data[0, 0] = np.nan
        np.savetxt(path, data)
        imputed, _, meta = read_data(path)
        assert meta['missing_by_feature'][0] == 1
        assert imputed[0, 0] == np.median(x[1:, 0])
    print('Все проверки пройдены: данные, PCA, потери, реконструкция, пропуски.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--scale', choices=['standard', 'center'], default='standard')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    try:
        x, labels, info = read_data(args.data)
        z, mean, divisor = preprocess(x, args.scale)
        results, projections = analyze(z)
    except (ValueError, OSError, ArithmeticError) as exc:
        parser.error(str(exc))
    print(f'Объектов: {len(x)}; признаков: {x.shape[1]}; классы: {info["class_counts"]}')
    print(f'Заполнено пропусков: {sum(info["missing_by_feature"])}; шкала: {args.scale}')
    print('\nКомпонента  Собств. значение   Доля, %   Накопленная доля, %')
    cumulative = 0.
    for i, (value, ratio) in enumerate(zip(
            results['eigenvalues'], results['ratios_percent']), 1):
        cumulative += ratio
        print(f'PC{i:<8} {value:15.9f} {ratio:10.6f} {cumulative:20.6f}')
    for k, r in results['dimensions'].items():
        print(f'\nk={k}: сохранено {r["retained_percent"]:.6f}%; '
              f'потери {r["lost_percent"]:.6f}%; SSE={r["reconstruction_sse"]:.6f}')
        print(f'sklearn: сохранено {r["sklearn_retained_percent"]:.6f}%; '
              f'потери {r["sklearn_lost_percent"]:.6f}%')
        print(f'Расхождение проекторов: {r["projector_max_error"]:.3e}; '
              f'координат после выбора знаков: {r["scores_max_error"]:.3e}')
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        import sklearn
        results.update(data=info, scale=args.scale, mean=mean.tolist(),
                       divisor=divisor.tolist(),
                       versions={'numpy': np.__version__, 'pandas': pd.__version__,
                                 'scikit-learn': sklearn.__version__})
        (args.out / 'results.json').write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    visualize(results, projections, labels, args.out, not args.no_show)
    if args.out:
        print(f'Графики и JSON сохранены: {args.out.resolve()}')


if __name__ == '__main__':
    main()
