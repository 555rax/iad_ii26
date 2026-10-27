import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Настройка стиля графиков
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

# 1. Загрузка данных
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0 - malignant, 1 - benign

# 2. Стандартизация и разделение данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# 3. Инициализация моделей
models = {
    "Одиночное дерево": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Случайный лес": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
        n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE),
    "CatBoost": CatBoostClassifier(iterations=300, depth=4, learning_rate=0.1, random_seed=RANDOM_STATE, verbose=0),
}

# Для кросс-валидации
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = []
matrices = {}

print("\n" + "=" * 50)
print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛЕЙ")
print("=" * 50)

# 4. Обучение, оценка и кросс-валидация
for name, model in models.items():
    # Кросс-валидация на обучающей выборке
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    # Обучение на полной обучающей выборке
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time

    y_pred = model.predict(X_test)

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    matrices[name] = cm

    # Метрики для класса 0 (malignant)
    precision = precision_score(y_test, y_pred, pos_label=0)
    recall = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred, pos_label=0)
    acc = accuracy_score(y_test, y_pred)

    results.append({
        "Модель": name,
        "Accuracy test": acc,
        "Precision (malignant)": precision,
        "Recall (malignant)": recall,
        "F1-score (malignant)": f1,
        "CV mean": cv_scores.mean(),
        "CV std": cv_scores.std(ddof=1),
        "Время обучения (с)": fit_time
    })

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["malignant (0)", "benign (1)"], digits=3))

# Сводная таблица
df_results = pd.DataFrame(results).sort_values(by="Recall (malignant)", ascending=False)
print("\n" + "=" * 80)
print("СВОДНАЯ ТАБЛИЦА МЕТРИК:")
print("=" * 80)
print(df_results.round(4).to_string(index=False))

# --- ГРАФИК 1: СРАВНЕНИЕ ACCURACY (КАК В ПРИМЕРЕ) ---
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(df_results))
width = 0.35

# Столбцы для Test Accuracy
ax.bar(x_pos - width / 2, df_results["Accuracy test"], width, label="Test Accuracy", color="#1f77b4")

# Столбцы для CV mean с усами погрешности (std)
ax.bar(x_pos + width / 2, df_results["CV mean"], width, yerr=df_results["CV std"],
       capsize=5, label="CV на train", color="#ff7f0e")

ax.set_xticks(x_pos)
ax.set_xticklabels(df_results["Модель"], rotation=15)
ax.set_ylim(0.8, 1.02)  # Ограничим ось Y, чтобы разница была нагляднее
ax.set_ylabel("Accuracy")
ax.set_title("Сравнение моделей на Breast Cancer (Accuracy)")
ax.legend(loc="lower left")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("cancer_accuracy_comparison.png", dpi=150)
plt.show()

# --- ГРАФИК 2: МАТРИЦЫ ОШИБОК ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Сортируем для красоты (по алфавиту или как в словаре)
for i, (name, cm) in enumerate(matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                xticklabels=['Malignant (0)', 'Benign (1)'],
                yticklabels=['Malignant (0)', 'Benign (1)'])
    axes[i].set_title(name)
    axes[i].set_xlabel('Предсказано')
    axes[i].set_ylabel('Истинное значение')

axes[-1].axis('off')  # Убираем пустой 6-й график
plt.tight_layout()
plt.savefig("cancer_confusion_matrices.png", dpi=150)
plt.show()

best_model_name = df_results.iloc[0]["Модель"]
print(f"\nНаиболее надежная модель (по Recall для malignant): {best_model_name}")