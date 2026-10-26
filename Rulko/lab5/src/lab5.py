
from pathlib import Path
import argparse, base64, csv, hashlib, io, json, platform, time, zlib
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.base import clone
import catboost, xgboost
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

DATA = 'c$|G!+iv4P2z~Eo_}Q{F7iPxau~OfoZq$twwN)DJs(-&@z=;hfd8=VKz~MkTd>_9K=jnX>c)$J`#z8*+oqqGz*YUb_%IE6*d$^pA*Pr2qrZA?7#xUh^isSI>c>Vl73@>?_#{{gkzmDCWjJ<Sw=yuxKsauOfq5hpZe&}fZPlJ->ecGdBhn>4EG76}CttHs372q_*FWxTbVQWon4F<c73qmw~)-kV@XViZ(_RTEP9w+q-%vZASo8TvlpXGhD@2jgvc<*xpualq<;%m0Eb{x{3*<GJe`!s4{dw!xiFd?3F0s2C`$373F4C?myTuG*@2&PTimEDc`z6D{g-{f8k>t5{}?6k$8z5@x^Gu+b^rSnygj%K5a+Xg_@*=ZEbX}3MWy(3lE{<Tax^ZQ=P>+$k^diy+oUWS*VO=XpiamuVR@0D?q(v+DH6Suhf_VR>WTvaEb$iy8;t!3|A%7|QVZ_&mTzFU$p*^IF}tDXl#K(AEhXJeLAEdsWzbg;OzZA%DR0-%|+zJ%&4TwBny1|kz4&vGCmpTppbiBctwTZ$p&12xW(iy9m1vEi8~28CL=#W6S0OHhbQ2v*t$%oJ|FX{FBDIUR`)t()Hbh?hl~1<Qd{dJR#wI;)Rs8|m4`*+qJ6O<aC7b@@5ZdU75JsiGMq+1Ck7Z&g`9E|S4MautPiV+{;hkfxi&`6#(5V8cbURaep#Qg7R80ePr1j*~TcAU$HylKyA%;D!Oh)*B{EHoTrL@28K`+u^PT8Tq{PRi=yh#D$$l1~-qyR5s%yj%vK^d%R_$au@e_az;rdR85rRDolvYRXL7smth+h`L-41%+=Wa%%|~W<s&&h3ja=u!h&H@>o91f_dw>BT_&kPmH=@`pey+wQ98H$%CRWSra|P_>+Wtd&00luF{1Cr_Y-B(Du6Po-dsPeMTL?pfN8}M=_zh*s`bf2qg+{;_iE!@s+nRuS^GlUFe~#JYG%!a_-MLcmixf-qT4|-b5`wIZ6rRok_~ApK;_6aMI6o=lfj->^`TsAoUifORpBjS#Q~M>6k^%ppHEhUXGFHo=b|;nX-YWEp!S(#f@SJIteNDE'
SHA256 = '119e30db8a4ea8b33723603743591a5f8229684e6236d89ef1966a72d7293607'
FEATURES = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
CLASSES = ['Setosa', 'Versicolor', 'Virginica']
NAMES = ['Дерево', 'Случайный лес', 'AdaBoost', 'CatBoost', 'XGBoost']

def load_data(path=None):
    raw = Path(path).read_bytes() if path else zlib.decompress(base64.b85decode(DATA))
    digest = hashlib.sha256(raw).hexdigest()
    if path is None and digest != SHA256:
        raise ValueError('Нарушена целостность встроенного CSV')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if reader.fieldnames != FEATURES + ['variety']:
        raise ValueError('Ожидаются столбцы: ' + ', '.join(FEATURES + ['variety']))
    rows = list(reader)
    X = np.array([[float(row[f]) for f in FEATURES] for row in rows], dtype=float)
    y = np.array([CLASSES.index(row['variety']) for row in rows], dtype=int)
    if X.shape != (150, 4) or not np.isfinite(X).all() or not np.all(X > 0):
        raise ValueError('Ожидается полный Iris: 150 строк и 4 положительных числовых признака')
    if np.bincount(y, minlength=3).tolist() != [50, 50, 50]:
        raise ValueError('В Iris должно быть по 50 объектов каждого класса')
    return X, y, digest

def models():
    # Параметры фиксированы заранее; тест не используется для их подбора.
    ada_args = dict(estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                    n_estimators=200, learning_rate=.5, random_state=42)
    # В старых sklearn явно выбираем SAMME; в новых это единственный алгоритм.
    import inspect
    if 'algorithm' in inspect.signature(AdaBoostClassifier).parameters:
        ada_args['algorithm'] = 'SAMME'
    return [DecisionTreeClassifier(max_depth=3, random_state=42),
            RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                   random_state=42, n_jobs=1),
            AdaBoostClassifier(**ada_args),
            CatBoostClassifier(iterations=300, depth=4, learning_rate=.05,
                               loss_function='MultiClass', random_seed=42,
                               thread_count=1, verbose=False, allow_writing_files=False),
            XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.05,
                          objective='multi:softprob', num_class=3, eval_metric='mlogloss',
                          tree_method='hist', subsample=1., colsample_bytree=1.,
                          random_state=42, n_jobs=1)]

def split(y):
    return train_test_split(np.arange(len(y)), test_size=.3, stratify=y, random_state=42)

def self_test():
    X,y,_=load_data(); tr,te=split(y)
    assert len(tr)==105 and len(te)==45 and not set(tr)&set(te)
    assert sorted(np.r_[tr,te])==list(range(150))
    assert np.bincount(y[tr]).tolist()==[35]*3 and np.bincount(y[te]).tolist()==[15]*3
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X[tr],y[tr]))
    held=[]
    for a,b in folds:
        assert not set(tr[a])&set(te) and not set(a)&set(b)
        held.extend(b)
    assert sorted(held)==list(range(105))
    for m in models():
        a=clone(m).fit(X[tr],y[tr]); b=clone(m).fit(X[tr],y[tr])
        pred=np.asarray(a.predict(X[te])).reshape(-1).astype(int)
        assert np.array_equal(pred,np.asarray(b.predict(X[te])).reshape(-1))
        prob=a.predict_proba(X[te]); assert prob.shape==(45,3)
        assert np.isfinite(prob).all() and np.allclose(prob.sum(axis=1),1,atol=1e-6)
        cm=confusion_matrix(y[te],pred,labels=[0,1,2])
        assert cm.sum()==45 and np.isclose(np.trace(cm)/45,accuracy_score(y[te],pred))
    print('Проверки CSV, разбиения, CV, вероятностей, метрик и повторяемости пройдены.')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',type=Path); ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results_lab5')
    ap.add_argument('--no-show',action='store_true'); ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test: self_test(); return
    import matplotlib
    if args.no_show: matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    X,y,digest=load_data(args.csv); tr,te=split(y); out=args.out; out.mkdir(parents=True,exist_ok=True)
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X[tr],y[tr]))
    results=[]; fitted=[]
    for name,model in zip(NAMES,models()):
        scores=[]
        for a,b in folds:
            fold=clone(model).fit(X[tr[a]],y[tr[a]])
            scores.append(accuracy_score(y[tr[b]],np.asarray(fold.predict(X[tr[b]])).reshape(-1)))
        start=time.perf_counter(); model.fit(X[tr],y[tr]); elapsed=time.perf_counter()-start
        pred=np.asarray(model.predict(X[te])).reshape(-1).astype(int)
        train_pred=np.asarray(model.predict(X[tr])).reshape(-1).astype(int)
        cm=confusion_matrix(y[te],pred,labels=[0,1,2])
        rec=dict(name=name,accuracy=accuracy_score(y[te],pred),train_accuracy=accuracy_score(y[tr],train_pred),
                 macro_f1=f1_score(y[te],pred,average='macro'),correct=int(np.trace(cm)),errors=int((pred!=y[te]).sum()),
                 cm=cm.tolist(),cv_scores=scores,cv_mean=float(np.mean(scores)),cv_std=float(np.std(scores,ddof=1)),
                 seconds=elapsed,predictions=pred.tolist(),
                 report=classification_report(y[te],pred,target_names=CLASSES,output_dict=True,zero_division=0))
        results.append(rec); fitted.append(model)
        print(f'{name:15} test accuracy={rec["accuracy"]:.4f}; F1={rec["macro_f1"]:.4f}; CV={rec["cv_mean"]:.4f} ± {rec["cv_std"]:.4f}',flush=True)
    summary=dict(source=str(args.csv) if args.csv else 'embedded iris.csv',sha256=digest,
                 versions=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,catboost=catboost.__version__,xgboost=xgboost.__version__),
                 train_indices=tr.tolist(),test_indices=te.tolist(),
                 duplicate_extra_rows=int(len(X)-len(np.unique(np.c_[X,y],axis=0))),
                 duplicate_across_split=int(sum(any(np.array_equal(x,t) for t in X[tr]) for x in X[te])),
                 statistics=dict(min=X.min(0).tolist(),max=X.max(0).tolist(),mean=X.mean(0).tolist(),std=X.std(0,ddof=1).tolist()),models=results,
                 tree_depth=int(fitted[0].get_depth()),tree_leaves=int(fitted[0].get_n_leaves()),tree_importances=fitted[0].feature_importances_.tolist())
    (out/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        fields=['name','accuracy','train_accuracy','macro_f1','correct','errors','cv_mean','cv_std','seconds']
        w=csv.DictWriter(fp,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in results])
    with (out/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        w=csv.writer(fp);w.writerow(['csv_line','true']+NAMES)
        for j,idx in enumerate(te):w.writerow([int(idx)+2,CLASSES[y[idx]]]+[CLASSES[r['predictions'][j]] for r in results])
    (out/'tree_rules.txt').write_text(export_text(fitted[0],feature_names=FEATURES),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    def save(fig,name):fig.savefig(out/name,dpi=190,bbox_inches='tight')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for ax,(a,b) in zip(axes,[(0,1),(2,3)]):
        for c,label in enumerate(CLASSES):ax.scatter(X[y==c,a],X[y==c,b],label=label,s=22,alpha=.75)
        ax.set(xlabel=FEATURES[a]+' (см)',ylabel=FEATURES[b]+' (см)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);save(fig,'data.png')
    fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained');xx=np.arange(5)
    ax.bar(xx-.18,[r['accuracy'] for r in results],.36,label='Test')
    ax.bar(xx+.18,[r['cv_mean'] for r in results],.36,yerr=[r['cv_std'] for r in results],capsize=3,label='CV на train ± SD')
    ax.set(xticks=xx,xticklabels=NAMES,ylim=(0,1.12),ylabel='Accuracy');ax.legend();ax.grid(axis='y',alpha=.2)
    for i,r in enumerate(results):ax.text(i-.18,.03,f'{r["correct"]}/45',ha='center',color='white')
    save(fig,'comparison.png')
    fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
    for ax,r in zip(axes.flat,results):
        cm=np.array(r['cm']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=15)
        ax.set(title=r['name'],xticks=range(3),yticks=range(3),xticklabels=['S','Ve','Vi'],yticklabels=['S','Ve','Vi'],xlabel='Прогноз',ylabel='Истинный класс')
        for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>7 else 'black')
    axes.flat[-1].axis('off');save(fig,'confusion.png')
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
    plot_tree(fitted[0],feature_names=FEATURES,class_names=CLASSES,filled=True,rounded=True,precision=2,fontsize=12,ax=ax)
    save(fig,'tree.png')
    print('Результаты:',out.resolve())
    if not args.no_show:plt.show()
    plt.close('all')

if __name__=='__main__':main()
