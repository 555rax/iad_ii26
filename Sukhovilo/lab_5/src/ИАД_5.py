"""ИАД ЛР №5 вариант 5: классификация грибов пятью деревообразными моделями."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/mushrooms.csv');target=(df['class']=='p').astype(int)
features=df.drop(columns=['class'])
# '?' = реальное неизвестное значение в отдельных категориальных признаках
X_tr,X_te,y_tr,y_te=train_test_split(features,target,test_size=.25,random_state=42,stratify=target)
encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
Xtrain=encoder.fit_transform(X_tr);Xtest=encoder.transform(X_te)
models={
 'Decision Tree':DecisionTreeClassifier(random_state=42,max_depth=12),
 'Random Forest':RandomForestClassifier(n_estimators=120,random_state=42,n_jobs=2),
 'AdaBoost':AdaBoostClassifier(n_estimators=120,learning_rate=.7,random_state=42),
 'CatBoost':CatBoostClassifier(iterations=160,learning_rate=.09,depth=5,random_seed=42,verbose=False,thread_count=2),
 'XGBoost':XGBClassifier(n_estimators=160,max_depth=5,learning_rate=.09,random_state=42,eval_metric='logloss',n_jobs=2)
}
metrics={}
for name,model in models.items():
 start=time.perf_counter();model.fit(Xtrain,y_tr);timing=time.perf_counter()-start
 pred=model.predict(Xtest)
 metrics[name]={'accuracy':float(accuracy_score(y_te,pred)),'precision_poisonous':float(precision_score(y_te,pred,zero_division=0)),
  'recall_poisonous':float(recall_score(y_te,pred,zero_division=0)),'f1_poisonous':float(f1_score(y_te,pred,zero_division=0)),
  'false_negatives':int(((y_te==1)&(pred==0)).sum()),'confusion_matrix':confusion_matrix(y_te,pred).tolist(),'training_seconds':round(timing,3)}
 fig,ax=plt.subplots(figsize=(4.5,4));ConfusionMatrixDisplay.from_predictions(y_te,pred,display_labels=['Edible','Poisonous'],ax=ax,colorbar=False,cmap='Blues');ax.set_title(name);fig.tight_layout();fig.savefig(OUT/f'confusion_{name.lower().replace(" ","_")}.png',dpi=140);plt.close(fig)
 print(name,metrics[name])
(OUT/'metrics.json').write_text(json.dumps({'dataset_rows':len(df),'train_size':len(y_tr),'test_size':len(y_te),'encoded_features':Xtrain.shape[1],'models':metrics},indent=2,ensure_ascii=False),encoding='utf-8')
fig,ax=plt.subplots(figsize=(9,4));names=list(metrics);vals=[metrics[n]['recall_poisonous'] for n in names];ax.bar(names,vals);ax.set_ylim(0,1.07);ax.set_ylabel('Recall (poisonous)');ax.tick_params(axis='x',labelrotation=15);fig.tight_layout();fig.savefig(OUT/'model_comparison.png',dpi=150);plt.close(fig)
