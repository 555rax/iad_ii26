"""ИАД, ЛР №2, вариант 11: WDBC + Autoencoder, t-SNE, PCA."""
from pathlib import Path
import json,random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
SEED=42
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/wdbc.csv')
# Второй столбец исходного WDBC — диагноз M/B, ID исключается.
y=(df['Diagnosis']=='M').astype(int).values
features=df.drop(columns=['ID','Diagnosis'])
X=StandardScaler().fit_transform(features).astype(np.float32)
# Масштабирование к [0,1] для декодера с Sigmoid.
from sklearn.preprocessing import MinMaxScaler
X=MinMaxScaler().fit_transform(X).astype(np.float32)
class AE(nn.Module):
 def __init__(self,d,latent):
  super().__init__();self.enc=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,latent));self.dec=nn.Sequential(nn.Linear(latent,32),nn.ReLU(),nn.Linear(32,d),nn.Sigmoid())
 def forward(self,x):return self.dec(self.enc(x))

def train(latent,epochs=70):
 model=AE(X.shape[1],latent);opt=torch.optim.Adam(model.parameters(),lr=.002)
 loader=torch.utils.data.DataLoader(torch.from_numpy(X),batch_size=256,shuffle=True)
 losses=[]
 for ep in range(epochs):
  total=0
  for z in loader:
   opt.zero_grad();loss=nn.functional.mse_loss(model(z),z);loss.backward();opt.step();total+=float(loss)*len(z)
  losses.append(total/len(X))
 with torch.no_grad():out=model.enc(torch.from_numpy(X)).numpy()
 return model,out,losses

def graph(data,kind,k,suffix='',labels=None):
 if labels is None: labels=y
 fig=plt.figure(figsize=(7.2,5.6));ax=fig.add_subplot(111,projection='3d' if k==3 else None)
 for digit in range(2):
  v=data[labels==digit]
  if k==3:ax.scatter(*v.T,s=3,alpha=.55,label=('Benign' if digit==0 else 'Malignant'))
  else:ax.scatter(v[:,0],v[:,1],s=4,alpha=.5,label=('Benign' if digit==0 else 'Malignant'))
 ax.set(xlabel='Component 1',ylabel='Component 2',title=f'{kind}: {k}D {suffix}')
 if k==3:ax.set_zlabel('Component 3')
 ax.legend(title='Diagnosis',ncol=2,fontsize=7,loc='best')
 fig.tight_layout();fig.savefig(OUT/f'{kind.lower().replace(" ","_")}_{k}d{suffix}.png',dpi=135);plt.close(fig)

metrics={'dataset':'UCI WDBC, diagnosis (second column)','samples':len(X),'features':X.shape[1],'classes':np.unique(y).tolist(),'autoencoders':{},'pca':{},'tsne':{},'tsne_sample_size':{'2D':569,'3D':350}}
for k in (2,3):
 model,z,h=train(k);graph(z,'Autoencoder',k)
 torch.save(model.state_dict(),OUT/f'autoencoder_{k}d.pt')
 metrics['autoencoders'][str(k)]={'epochs':len(h),'initial_mse':h[0],'final_mse':h[-1]}
 plt.figure(figsize=(6,3));plt.plot(h);plt.title(f'Autoencoder {k}D — training MSE');plt.xlabel('Epoch');plt.ylabel('MSE');plt.tight_layout();plt.savefig(OUT/f'ae_training_{k}d.png',dpi=130);plt.close()
 # PCA from Lab 1: centered covariance and explicit eigenvectors (NumPy).
 centered=X-X.mean(axis=0,keepdims=True)
 covariance=np.cov(centered,rowvar=False)
 eigenvalues,eigenvectors=np.linalg.eigh(covariance)
 order=np.argsort(eigenvalues)[::-1]
 eigenvalues=eigenvalues[order];eigenvectors=eigenvectors[:,order]
 projected_numpy=centered@eigenvectors[:,:k]
 graph(projected_numpy,'PCA NumPy',k)
 model_pca=PCA(n_components=k,svd_solver='full')
 projected_sklearn=model_pca.fit_transform(X)
 graph(projected_sklearn,'PCA',k)
 diff=float(np.max(np.abs(eigenvalues[:k]-model_pca.explained_variance_)))
 assert diff<1e-6,(k,diff)
 metrics['pca'][str(k)]={'retained_variance_percent':float(100*eigenvalues[:k].sum()/eigenvalues.sum()),
                        'loss_percent':float(100*(1-eigenvalues[:k].sum()/eigenvalues.sum())),
                        'numpy_sklearn_eigenvalue_max_diff':diff}

 # t-SNE сравниваем при нескольких значениях perplexity.
 for p in (20,40,60):
  # Случайная воспроизводимая подвыборка для вычислительно дорогого t-SNE
  chosen=np.random.default_rng(SEED).choice(len(X),size=(350 if k==3 else len(X)),replace=False)
  ts=TSNE(n_components=k,perplexity=p,init='pca',learning_rate='auto',random_state=SEED,max_iter=500,method='barnes_hut' if k==2 else 'exact')
  q=ts.fit_transform(X[chosen])
  graph(q,'TSNE',k,f'_p{p}',labels=y[chosen])
  metrics['tsne'][f'{k}D_perplexity_{p}']={'kl_divergence':float(ts.kl_divergence_)}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(metrics,ensure_ascii=False,indent=2))
