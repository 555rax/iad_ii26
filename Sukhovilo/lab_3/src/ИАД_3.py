"""ИАД ЛР №3, вариант 5: обучение DNN с AE-предобучением и без него, NSP и optdigits."""
from pathlib import Path
from NN_common import datasets,train,report
ROOT=Path(__file__).resolve().parent
ans={}
for name,(X,y,tr,te) in datasets(ROOT).items():
 ans[name]=[train(X,y,tr,te,method,ROOT,name) for method in ['baseline','autoencoder']]
report(ans,ROOT)
