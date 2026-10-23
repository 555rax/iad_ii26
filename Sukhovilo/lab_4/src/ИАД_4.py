"""ИАД ЛР №4, вариант 5: сравнение baseline, AE, RBM на CTG (NSP) и optdigits."""
from pathlib import Path
from NN_common import datasets,train,report
ROOT=Path(__file__).resolve().parent
ans={}
for name,(X,y,tr,te) in datasets(ROOT).items():
 ans[name]=[train(X,y,tr,te,method,ROOT,name) for method in ['baseline','autoencoder','RBM']]
report(ans,ROOT)
