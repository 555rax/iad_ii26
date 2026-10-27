from pathlib import Path
import hashlib
import io
import json
import platform
import time
import urllib.request
import zipfile
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report, confusion_matrix,
    precision_recall_curve)
import catboost
from catboost import CatBoostClassifier
import xgboost
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
SEED = 42
COLUMNS = ['age','workclass','fnlwgt','education','education-num','marital-status',
           'occupation','relationship','race','sex','capital-gain','capital-loss',
           'hours-per-week','native-country','income']
NUMERIC = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
CATEGORICAL = [c for c in COLUMNS[:-1] if c not in NUMERIC]
PARAMETERS = {
    'DecisionTree': dict(max_depth=8, min_samples_leaf=20, criterion='gini', random_state=SEED),
    'RandomForest': dict(n_estimators=200, max_depth=None, min_samples_leaf=2,
                         max_features='sqrt', n_jobs=2, random_state=SEED),
    'AdaBoost': dict(n_estimators=200, learning_rate=.5, random_state=SEED),
    'CatBoost': dict(iterations=500, depth=6, learning_rate=.05, loss_function='Logloss',
                    random_seed=SEED, thread_count=2, verbose=False, allow_writing_files=False),
    'XGBoost': dict(n_estimators=400, max_depth=4, learning_rate=.05, subsample=.8,
                   colsample_bytree=.8, objective='binary:logistic', eval_metric='logloss',
                   tree_method='hist', n_jobs=2, random_state=SEED)
}


def load_data():
    folder = ROOT / 'data'
    folder.mkdir(exist_ok=True)
    if not all((folder / n).exists() for n in ['adult.data', 'adult.test']):
        with urllib.request.urlopen('https://archive.ics.uci.edu/static/public/2/adult.zip', timeout=120) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        for name in ['adult.data','adult.test','adult.names']:
            (folder/name).write_bytes(z.read(name))
    frames = []
    for name in ['adult.data', 'adult.test']:
        df = pd.read_csv(folder/name, names=COLUMNS, skipinitialspace=True, na_values='?', comment='|')
        for column in CATEGORICAL + ['income']:
            df[column] = df[column].str.strip()
        df['income'] = df['income'].str.rstrip('.')
        assert df['income'].isin(['<=50K','>50K']).all()
        frames.append(df)
    return frames


def preprocessing():
    return ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), NUMERIC),
        ('category', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32))
        ]), CATEGORICAL)], sparse_threshold=0)


def main():
    train, test = load_data()
    xtrain, xtest = train.drop(columns='income'), test.drop(columns='income')
    ytrain = (train.income == '>50K').astype(int)
    ytest = (test.income == '>50K').astype(int)
    transformer = preprocessing()
    started = time.perf_counter()
    a = transformer.fit_transform(xtrain).astype(np.float32)
    b = transformer.transform(xtest).astype(np.float32)
    preparation_seconds = time.perf_counter() - started
    joblib.dump(transformer, OUT/'preprocessor.joblib', compress=3)
    cat_medians = xtrain[NUMERIC].median()
    cat_train, cat_test = xtrain.copy(), xtest.copy()
    for df in [cat_train, cat_test]:
        df[NUMERIC] = df[NUMERIC].fillna(cat_medians)
        df[CATEGORICAL] = df[CATEGORICAL].fillna('Missing')
    metadata = dict(seed=SEED, threshold=.5, train_rows=len(train), test_rows=len(test),
        numeric=NUMERIC, categorical=CATEGORICAL, encoded_features=a.shape[1],
        class_counts={'train':ytrain.value_counts().sort_index().to_dict(),
                      'test':ytest.value_counts().sort_index().to_dict()},
        missing={'train':xtrain.isna().sum().to_dict(),'test':xtest.isna().sum().to_dict()},
        medians=cat_medians.to_dict(), preprocessing_seconds=preparation_seconds,
        parameters=PARAMETERS, versions=dict(python=platform.python_version(),numpy=np.__version__,
        pandas=pd.__version__,sklearn=sklearn.__version__,catboost=catboost.__version__,xgboost=xgboost.__version__),
        sha256={n:hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest() for n in ['adult.data','adult.test']})
    (OUT/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame({'source_file':'adult.data','source_row':np.arange(len(train))}).to_csv(OUT/'train_indices.csv',index=False)
    pd.DataFrame({'source_file':'adult.test','source_row':np.arange(len(test))}).to_csv(OUT/'test_indices.csv',index=False)
    models = {
        'DecisionTree':DecisionTreeClassifier(**PARAMETERS['DecisionTree']),
        'RandomForest':RandomForestClassifier(**PARAMETERS['RandomForest']),
        'AdaBoost':AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1,random_state=SEED),**PARAMETERS['AdaBoost']),
        'CatBoost':CatBoostClassifier(**PARAMETERS['CatBoost']),
        'XGBoost':XGBClassifier(**PARAMETERS['XGBoost'])}
    results, probabilities, matrices = [], {}, {}
    for name, model in models.items():
        print('Training', name, flush=True)
        tr, te = (cat_train,cat_test) if name=='CatBoost' else (a,b)
        started=time.perf_counter()
        if name=='CatBoost': model.fit(tr,ytrain,cat_features=CATEGORICAL)
        else: model.fit(tr,ytrain)
        fit_seconds=time.perf_counter()-started
        started=time.perf_counter()
        probability=model.predict_proba(te)[:,1]
        prediction=(probability>=.5).astype(int)
        predict_seconds=time.perf_counter()-started
        cm=confusion_matrix(ytest,prediction,labels=[0,1])
        tn,fp,fn,tp=cm.ravel()
        row=dict(model=name,precision=precision_score(ytest,prediction,zero_division=0),
            recall=recall_score(ytest,prediction),f1=f1_score(ytest,prediction),
            accuracy=accuracy_score(ytest,prediction),roc_auc=roc_auc_score(ytest,probability),
            average_precision=average_precision_score(ytest,probability),
            TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
            predicted_positive=int((prediction==1).sum()),
            fit_seconds=fit_seconds,predict_seconds=predict_seconds)
        results.append(row); probabilities[name]=probability;matrices[name]=cm
        print(json.dumps(row),flush=True)
        pd.DataFrame(results).to_csv(OUT/'metrics.csv',index=False)
        pd.DataFrame({'source_row':np.arange(len(test)),'actual':ytest,'predicted':prediction,
                      'probability_gt50k':probability}).to_csv(OUT/f'{name}_predictions.csv',index=False)
        (OUT/f'{name}_classification_report.txt').write_text(classification_report(ytest,prediction,
            target_names=['<=50K','>50K'],digits=5),encoding='utf-8')
        pd.DataFrame(cm,index=['actual_<=50K','actual_>50K'],columns=['predicted_<=50K','predicted_>50K']).to_csv(OUT/f'{name}_confusion.csv')
        joblib.dump(model,OUT/f'{name}.joblib',compress=3)
    plots(pd.DataFrame(results),ytest,probabilities,matrices,models['DecisionTree'],transformer)
    from report import make_report
    make_report()


def plots(metrics, truth, probabilities, matrices, tree, transformer):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,ax=plt.subplots(figsize=(9,3.4),layout='constrained')
    x=np.arange(len(metrics));width=.25
    for offset,key,color in [(-1,'precision','#34739e'),(0,'recall','#d78940'),(1,'f1','#47856a')]:
        bars=ax.bar(x+offset*width,metrics[key],width,label=key,color=color)
        ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=2)
    ax.set(xticks=x,xticklabels=metrics.model,ylim=(0,1.05),ylabel='Значение метрики',title='Класс >50K, порог 0.5')
    ax.legend(loc='upper right',ncol=3,fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
    for ax,(name,cm) in zip(axes.flat,matrices.items()):
        ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        ax.set(xticks=[0,1],xticklabels=['<=50K','>50K'],yticks=[0,1],yticklabels=['<=50K','>50K'],title=name,xlabel='Прогноз',ylabel='Факт')
    axes.flat[-1].axis('off');fig.savefig(OUT/'confusion_matrices.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for name,prob in probabilities.items():
        precision,recall,_=precision_recall_curve(truth,prob)
        ax.plot(recall,precision,label=name)
    ax.axhline(np.mean(truth),color='gray',ls='--',label='Доля >50K')
    ax.set(xlabel='Recall',ylabel='Precision',title='Precision–Recall на тестовой выборке',xlim=(0,1),ylim=(0,1.02))
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.savefig(OUT/'pr_curves.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(16,6),layout='constrained')
    plot_tree(tree,feature_names=transformer.get_feature_names_out(),class_names=['<=50K','>50K'],
              max_depth=2,filled=True,rounded=True,fontsize=8,ax=ax)
    fig.savefig(OUT/'tree_top.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
