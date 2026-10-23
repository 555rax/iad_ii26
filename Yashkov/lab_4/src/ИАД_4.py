"""ЛР 4, вариант 11, Яшков П. А.: baseline vs AE vs RBM на Credit Approval и WDBC."""
from pathlib import Path
from NN_common import datasets,train,report
ROOT=Path(__file__).resolve().parent
result={}
for name,(X,y,tr,te) in datasets(ROOT).items():
 result[name]=[train(X,y,tr,te,m,ROOT,name) for m in ['baseline','autoencoder','RBM']]
report(result,ROOT)
