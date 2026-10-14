import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

print("Размер выборки:", X.shape)
print("Обучающая:", X_train.shape, "Тестовая:", X_test.shape)

models = {
    "Одиночное дерево": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Случайный лес": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
        n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        random_state=RANDOM_STATE),
    "CatBoost": CatBoostClassifier(
        iterations=300, depth=4, learning_rate=0.1,
        random_seed=RANDOM_STATE, verbose=0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rows = []
matrices = {}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    pred_train = model.predict(X_train)
    pred_test = np.asarray(model.predict(X_test)).reshape(-1)

    matrices[name] = confusion_matrix(y_test, pred_test)

    rows.append({
        "Модель": name,
        "Accuracy train": accuracy_score(y_train, pred_train),
        "Accuracy test": accuracy_score(y_test, pred_test),
        "Macro F1 test": f1_score(y_test, pred_test, average="macro"),
        "CV mean": cv_scores.mean(),
        "CV std": cv_scores.std(ddof=1),
        "Ошибок на тесте": int((pred_test != y_test).sum()),
        "Время обучения, с": fit_time,
    })

    print(f"\n===== {name} =====")
    print(classification_report(y_test, pred_test, digits=3))

results = pd.DataFrame(rows).sort_values("Accuracy test", ascending=False)
print("\nСводная таблица:")
print(results.round(4).to_string(index=False))
results.to_csv("digits_results.csv", index=False, encoding="utf-8-sig")

best = results.iloc[0]["Модель"]
print(f"\nЛучшая модель по accuracy на тесте: {best}")

fig, ax = plt.subplots(figsize=(9, 5))
x_pos = np.arange(len(results))
ax.bar(x_pos - 0.2, results["Accuracy test"], 0.4, label="Test")
ax.bar(x_pos + 0.2, results["CV mean"], 0.4, yerr=results["CV std"],
       capsize=3, label="CV на train")
ax.set_xticks(x_pos)
ax.set_xticklabels(results["Модель"], rotation=15)
ax.set_ylim(0.7, 1.02)
ax.set_ylabel("Accuracy")
ax.set_title("Сравнение моделей на Digits")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("digits_comparison.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (name, cm) in zip(axes.flat, matrices.items()):
    ax.imshow(cm, cmap="Blues")
    ax.set_title(name)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=7)
axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig("digits_confusion.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax, img, label in zip(axes.flat, digits.images, digits.target):
    ax.imshow(img, cmap="gray_r")
    ax.set_title(f"Цифра {label}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("digits_samples.png", dpi=150)
plt.close()