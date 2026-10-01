import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

RANDOM_STATE = 42
LOCAL_FILE = "diabetes.csv"
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data():
    if os.path.exists(LOCAL_FILE):
        df = pd.read_csv(LOCAL_FILE)
        if "Outcome" not in df.columns:
            df = pd.read_csv(LOCAL_FILE, header=None, names=COLUMNS)
    else:
        df = pd.read_csv(URL, header=None, names=COLUMNS)
    return df


df = load_data()

print("=" * 70)
print("1. ЗАГРУЗКА ДАННЫХ")
print("=" * 70)
print("Размер датасета:", df.shape)
print(df.head())
print("\nРаспределение классов:")
print(df["Outcome"].value_counts())
print(df["Outcome"].value_counts(normalize=True).round(3))
print("\nКоличество нулевых значений в признаках, где ноль невозможен:")
print((df[ZERO_AS_MISSING] == 0).sum())

df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)

X = df.drop(columns="Outcome")
y = df["Outcome"]
feature_names = X.columns.tolist()

print("\n" + "=" * 70)
print("2. РАЗДЕЛЕНИЕ 70/30 И СТАНДАРТИЗАЦИЯ")
print("=" * 70)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train = scaler.fit_transform(imputer.fit_transform(X_train_raw))
X_test = scaler.transform(imputer.transform(X_test_raw))

print("Обучающая выборка:", X_train.shape, "| доля класса 1: %.3f" % y_train.mean())
print("Тестовая выборка: ", X_test.shape, "| доля класса 1: %.3f" % y_test.mean())
print("Среднее обучающей после стандартизации:", np.round(X_train.mean(axis=0), 3))
print("Std обучающей после стандартизации:    ", np.round(X_train.std(axis=0), 3))


def build_models(balanced=False):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    cw = "balanced" if balanced else None
    spw = neg / pos if balanced else 1.0
    cb_w = "Balanced" if balanced else None

    return {
        "Одиночное дерево (без ограничений)": DecisionTreeClassifier(
            class_weight=cw, random_state=RANDOM_STATE
        ),
        "Одиночное дерево (max_depth=4)": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=5, class_weight=cw, random_state=RANDOM_STATE
        ),
        "Случайный лес": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            class_weight=cw,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=200,
            learning_rate=0.5,
            random_state=RANDOM_STATE,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=4,
            learning_rate=0.05,
            auto_class_weights=cb_w,
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate(models, title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    rows = []
    fitted = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        cv_recall = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="recall", n_jobs=1
        )

        rows.append(
            {
                "Модель": name,
                "Recall (класс 1)": recall_score(y_test, y_pred, pos_label=1),
                "Precision": precision_score(y_test, y_pred, pos_label=1),
                "F1": f1_score(y_test, y_pred, pos_label=1),
                "Accuracy": accuracy_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_proba),
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "CV recall (train, 5 fold)": cv_recall.mean(),
            }
        )

    result = pd.DataFrame(rows).set_index("Модель")
    result = result.sort_values("Recall (класс 1)", ascending=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(result.round(3))
    return result, fitted


results_base, fitted_base = evaluate(
    build_models(balanced=False),
    "3-4. ОБУЧЕНИЕ МОДЕЛЕЙ И ОЦЕНКА НА ТЕСТЕ (стандартные веса классов)",
)

results_bal, fitted_bal = evaluate(
    build_models(balanced=True),
    "5. ТЕ ЖЕ МОДЕЛИ С УЧЁТОМ ДИСБАЛАНСА КЛАССОВ (class_weight / scale_pos_weight)",
)

summary = pd.DataFrame(
    {
        "Recall (стандартные веса)": results_base["Recall (класс 1)"],
        "Recall (balanced)": results_bal["Recall (класс 1)"],
        "Precision (balanced)": results_bal["Precision"],
        "ROC-AUC (balanced)": results_bal["ROC-AUC"],
    }
).sort_values("Recall (balanced)", ascending=False)

print("\n" + "=" * 70)
print("СВОДНАЯ ТАБЛИЦА")
print("=" * 70)
print(summary.round(3))

results_base.round(4).to_csv("results_standard.csv", encoding="utf-8-sig")
results_bal.round(4).to_csv("results_balanced.csv", encoding="utf-8-sig")
summary.round(4).to_csv("results_summary.csv", encoding="utf-8-sig")

order = results_base.index.tolist()
x = np.arange(len(order))
width = 0.38

fig, ax = plt.subplots(figsize=(11, 5.5))
b1 = ax.bar(
    x - width / 2,
    results_base.loc[order, "Recall (класс 1)"],
    width,
    label="Стандартные веса",
)
b2 = ax.bar(
    x + width / 2,
    results_bal.loc[order, "Recall (класс 1)"],
    width,
    label="С учётом дисбаланса",
)
for bars in (b1, b2):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            "%.2f" % bar.get_height(),
            ha="center",
            fontsize=9,
        )
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=20, ha="right")
ax.set_ylabel("Recall (класс 1 — диабет)")
ax.set_ylim(0, 1.05)
ax.set_title("Сравнение моделей по recall на тестовой выборке")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("recall_comparison.png", dpi=150)
plt.show()

best_name = results_bal.index[0]
print("\nЛучшая модель по recall (balanced):", best_name)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, (name, model) in zip(axes.ravel(), fitted_bal.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(name, fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Нет", "Диабет"])
    ax.set_yticklabels(["Нет", "Диабет"])
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=13,
            )
plt.suptitle("Матрицы ошибок (модели с учётом дисбаланса классов)")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

importances = pd.DataFrame(index=feature_names)
for name in ["Случайный лес", "AdaBoost", "CatBoost", "XGBoost"]:
    model = fitted_bal[name]
    imp = np.asarray(model.feature_importances_, dtype=float)
    importances[name] = imp / imp.sum()

print("\nВажность признаков:")
print(importances.round(3))

importances.plot(kind="bar", figsize=(12, 5.5))
plt.ylabel("Нормированная важность")
plt.title("Важность признаков в ансамблевых моделях")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

print("\nГотово. Файлы сохранены рядом со скриптом:")
print("results_standard.csv, results_balanced.csv, results_summary.csv")
print("recall_comparison.png, confusion_matrices.png, feature_importance.png")