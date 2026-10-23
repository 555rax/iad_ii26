"""ИАД, ЛР №1, вариант 5. Запуск: python ИАД_1.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data'/'Wholesale customers data.csv')
assert df.isna().sum().sum()==0
labels=df['Region'].astype(int).to_numpy()
# Region исключительно для графиков. Channel рассматривается как категориальный признак.
X=pd.get_dummies(df.drop(columns='Region'),columns=['Channel'],dtype=float)
X_scaled=StandardScaler().fit_transform(X)
# Независимая реализация через собственные числа/векторы
cov=np.cov(X_scaled,rowvar=False)
vals,vecs=np.linalg.eig(cov)
order=np.argsort(vals.real)[::-1];vals=vals.real[order];vecs=vecs.real[:,order]
manual={k:X_scaled@vecs[:,:k] for k in (2,3)}
skmodels={k:PCA(n_components=k).fit(X_scaled) for k in (2,3)}
sk={k:skmodels[k].transform(X_scaled) for k in (2,3)}
loss={k:float(100*(1-vals[:k].sum()/vals.sum())) for k in (2,3)}
report={'rows':len(df),'features_after_encoding':list(X.columns),'eigenvalues':vals.tolist(),
        'loss_percent':{str(k):v for k,v in loss.items()},
        'explained_variance':{str(k):float(skmodels[k].explained_variance_ratio_.sum()) for k in (2,3)},
        'eigenvalues_match':bool(np.allclose(vals[:3],skmodels[3].explained_variance_))}
(OUT/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
colors=['#c7424b','#2b8a57','#376ad4']
for k in (2,3):
 fig=plt.figure(figsize=(12,5))
 for pos,(label,points) in enumerate([('NumPy eig',manual[k]),('sklearn PCA',sk[k])],start=1):
  ax=fig.add_subplot(1,2,pos,projection='3d' if k==3 else None)
  for i,cl in enumerate(sorted(np.unique(labels))):
   q=points[labels==cl]
   if k==3: ax.scatter(q[:,0],q[:,1],q[:,2],s=11,alpha=.68,color=colors[i],label=f'Region {cl}')
   else: ax.scatter(q[:,0],q[:,1],s=15,alpha=.68,color=colors[i],label=f'Region {cl}')
  ax.set(xlabel='PC1',ylabel='PC2',title=f'{label}: {k} components')
  if k==3: ax.set_zlabel('PC3')
  ax.legend(fontsize=8)
  if k==2: ax.grid(alpha=.3)
 fig.tight_layout();fig.savefig(OUT/f'pca_{k}d.png',dpi=170);plt.close(fig)
print('Samples:',len(df),'Features:',len(X.columns),'Class counts:',df.Region.value_counts().to_dict())
print('PCA loss: 2D %.2f%%, 3D %.2f%%'%(loss[2],loss[3]))
print('NumPy vs sklearn eigenvalues match:',report['eigenvalues_match'])
