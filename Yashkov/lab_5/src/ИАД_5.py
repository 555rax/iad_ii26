"""ИАД, лабораторная №5, вариант 11, Яшков П. А.: Bank Marketing."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/bank.csv')
assert 'deposit' in df.columns
X=df.drop(columns=['deposit']);y=df.deposit.astype(str).str.lower().eq('yes').astype(int)
cat=X.select_dtypes(include='object').columns.tolist();numeric=X.select_dtypes(exclude='object').columns.tolist()
Xt,Xs,yt,ys=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
pre=ColumnTransformer([
 ('n',Pipeline([('fill',SimpleImputer(strategy='median'))]),numeric),
 ('c',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)
],sparse_threshold=0)
train=pre.fit_transform(Xt);test=pre.transform(Xs)
models={
 'Decision Tree':DecisionTreeClassifier(max_depth=8,min_samples_leaf=8,random_state=42),
 'Random Forest':RandomForestClassifier(n_estimators=160,min_samples_leaf=2,random_state=42,n_jobs=2),
 'AdaBoost':AdaBoostClassifier(n_estimators=150,learning_rate=.65,random_state=42),
 'CatBoost':CatBoostClassifier(iterations=210,depth=5,learning_rate=.075,random_seed=42,verbose=False,thread_count=2),
 'XGBoost':XGBClassifier(n_estimators=210,max_depth=5,learning_rate=.075,subsample=.9,colsample_bytree=.9,random_state=42,n_jobs=2,eval_metric='logloss')}
metrics={}
for name,model in models.items():
 t=time.perf_counter();model.fit(train,yt);duration=time.perf_counter()-t
 p=model.predict(test); cm=confusion_matrix(ys,p,labels=[0,1]);
 metrics[name]={"accuracy":float(accuracy_score(ys,p)),"precision_yes":float(precision_score(ys,p,zero_division=0)),"recall_yes":float(recall_score(ys,p,zero_division=0)),"f1_yes":float(f1_score(ys,p,zero_division=0)),"confusion_matrix":cm.tolist(),"training_seconds":round(duration,3)}
 fig,ax=plt.subplots(figsize=(5.1,4));ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['No','Yes']).plot(ax=ax,cmap='Blues',colorbar=False);ax.set_title(name);fig.tight_layout();fig.savefig(OUT/f'confusion_{name.lower().replace(" ","_")}.png',dpi=140);plt.close(fig)
 print(f'{name}: F1(yes)={metrics[name]["f1_yes"]:.4f}, Precision={metrics[name]["precision_yes"]:.4f}, Recall={metrics[name]["recall_yes"]:.4f}')
result={'variant':11,'student':'Яшков П. А.','dataset':'Course Drive Bank Marketing (bank.csv, deposit)','rows':len(df),'train_rows':len(Xt),'test_rows':len(Xs),'features_encoded':int(train.shape[1]),'target':'deposit=yes','models':metrics}
(OUT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
fig,ax=plt.subplots(figsize=(9,4.3));names=list(metrics);vals=[metrics[n]['f1_yes'] for n in names];b=ax.bar(names,vals);ax.set_ylabel('F1 для deposit=yes');ax.set_ylim(0,1.0);ax.tick_params(axis='x',labelrotation=15)
for rect,val in zip(b,vals):ax.text(rect.get_x()+rect.get_width()/2,val+.012,f'{val:.3f}',ha='center',fontsize=9)
fig.tight_layout();fig.savefig(OUT/'model_comparison.png',dpi=170);plt.close(fig)
