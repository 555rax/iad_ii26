import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

SEED = 42
path = sys.argv[1] if len(sys.argv) > 1 else "mushrooms.csv"

df = pd.read_csv(path)
print(f"Размер данных: {df.shape}")
print("Пропуски:", int(df.isna().sum().sum()))
print("Баланс классов:\n", df["class"].value_counts().to_string(), "\n")

# Целевая переменная: 1 = ядовитый (p), 0 = съедобный (e)
y = (df["class"] == "p").astype(int)

X = pd.get_dummies(df.drop(columns="class").astype(str), dtype=int)
print(f"Признаков после One-Hot: {X.shape[1]}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)
print(f"train: {X_train.shape[0]}, test: {X_test.shape[0]}\n")

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    # В sklearn >= 1.6 для AdaBoost используется алгоритм SAMME
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=200, learning_rate=0.5, random_state=SEED,
    ),
}

try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=SEED, n_jobs=-1,
    )
except ImportError:
    print("[!] xgboost не установлен -> XGBoost пропущен (pip install xgboost)")

try:
    from catboost import CatBoostClassifier
    models["CatBoost"] = CatBoostClassifier(
        iterations=300, depth=4, learning_rate=0.1,
        random_seed=SEED, verbose=0,
    )
except ImportError:
    print("[!] catboost не установлен -> CatBoost пропущен (pip install catboost)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    rows.append({
        "Модель": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision (ядовитый)": precision_score(y_test, pred, pos_label=1, zero_division=0),
        "Recall (ядовитый)": recall_score(y_test, pred, pos_label=1, zero_division=0),
        "FN (ядовитый -> съедобный)": fn,
        "FP (съедобный -> ядовитый)": fp,
        "CV acc (5-fold, весь набор)": cross_val_score(model, X, y, cv=cv).mean(),
        "Время обучения, с": round(fit_time, 2),
    })

res = pd.DataFrame(rows).set_index("Модель")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print("\n=== Результаты на тестовой выборке ===")
print(res.round(4).to_string())
res.round(4).to_csv("results.csv", encoding="utf-8-sig")

imp = pd.Series(models["Random Forest"].feature_importances_, index=X.columns)
print("\nТоп-10 признаков (Random Forest):")
print(imp.sort_values(ascending=False).head(10).round(4).to_string())