import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

SEED = 42
PATH = sys.argv[1] if len(sys.argv) > 1 else "seeds_dataset.txt"


cols = ["area", "perimeter", "compactness", "kernel_length",
        "kernel_width", "asymmetry", "groove_length", "class"]
df = pd.read_csv(PATH, sep=r"\s+", header=None)
df = df.iloc[:, -8:]
df.columns = cols
df["class"] = df["class"].astype(int)
names = {1: "Kama", 2: "Rosa", 3: "Canadian"}

print("Размер данных:", df.shape)
print("Пропуски:", int(df.isna().sum().sum()))
print(df["class"].map(names).value_counts(), "\n")

X = df.drop(columns="class").values
y = df["class"].values - 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y)

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)


models = {
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=SEED),
        n_estimators=200, learning_rate=0.5, random_state=SEED),
}
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        eval_metric="mlogloss", random_state=SEED)
except ImportError:
    print("! xgboost не установлен: pip install xgboost")
try:
    from catboost import CatBoostClassifier
    models["CatBoost"] = CatBoostClassifier(
        iterations=300, depth=4, learning_rate=0.1,
        loss_function="MultiClass", random_seed=SEED, verbose=0)
except ImportError:
    print("! catboost не установлен: pip install catboost")

results, preds = {}, {}
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train_s, y_train)
    fit_time = time.time() - t0
    pred = np.asarray(model.predict(X_test_s)).ravel().astype(int)
    preds[name] = pred
    results[name] = accuracy_score(y_test, pred)
    print(f"{name:14s} accuracy = {results[name]:.4f}   (обучение {fit_time:.2f} с)")


res = pd.Series(results).sort_values(ascending=False)
print("\nСравнение моделей по accuracy:\n", res.round(4).to_string())
best = res.index[0]
print(f"\nЛучшая модель: {best} ({res.iloc[0]:.4f})")
print(f"\nclassification_report для {best}:\n",
      classification_report(y_test, preds[best],
                           target_names=[names[i] for i in (1, 2, 3)]))

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
print("Кросс-валидация (5 folds), средняя accuracy ± std:")
for name, model in models.items():
    sc = cross_val_score(make_pipeline(StandardScaler(), model), X, y, cv=skf)
    print(f"  {name:14s} {sc.mean():.4f} ± {sc.std():.4f}")


pca = PCA(n_components=2).fit(X_train_s)
Z_test = pca.transform(X_test_s)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
palette = ["tab:blue", "tab:orange", "tab:green"]
for ax, labels, title in [
        (axes[0], y_test, "Истинные классы (тест)"),
        (axes[1], preds[best], f"Предсказания: {best}")]:
    for k in range(3):
        m = labels == k
        ax.scatter(Z_test[m, 0], Z_test[m, 1], c=palette[k], label=names[k + 1],
                   edgecolor="k", s=55)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.grid(alpha=.3)
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
wrong = preds[best] != y_test
axes[1].scatter(Z_test[wrong, 0], Z_test[wrong, 1], facecolors="none",
                edgecolors="red", s=220, linewidths=2, label="Ошибка")
axes[1].legend()
plt.tight_layout()
plt.savefig("pca_best_model.png", dpi=150)
print("\nГрафик сохранён: pca_best_model.png")