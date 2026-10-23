"""ЛР 3, вариант 11, Яшков П. А.: baseline vs AE на Credit Approval и WDBC."""
from pathlib import Path
from NN_common import datasets,train,report
ROOT=Path(__file__).resolve().parent
result={}
for name,(X,y,tr,te) in datasets(ROOT).items():
 result[name]=[train(X,y,tr,te,m,ROOT,name) for m in ['baseline','autoencoder']]
report(result,ROOT)
