
from pathlib import Path
import subprocess
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

USE_GPU = True

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА — ВАРИАНТ 7")
print("Прогнозирование оттока клиентов")
print("Telco Customer Churn")
print("=" * 60)


def check_nvidia_gpu():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader"
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True
        )

        gpu_names = result.stdout.strip()

        if gpu_names:
            print("\nОбнаружена видеокарта:")
            print(gpu_names)
            return True

        return False

    except (FileNotFoundError, subprocess.SubprocessError):
        print("\nNVIDIA GPU не обнаружена.")
        return False


print("\n1. ПРОВЕРКА CUDA")

nvidia_available = check_nvidia_gpu()

try:
    catboost_gpu_count = get_gpu_device_count()
except Exception:
    catboost_gpu_count = 0

catboost_use_gpu = (
    USE_GPU
    and nvidia_available
    and catboost_gpu_count > 0
)

xgboost_use_gpu = USE_GPU and nvidia_available

print(
    "CatBoost:",
    "GPU" if catboost_use_gpu else "CPU"
)

print(
    "XGBoost:",
    "GPU (проверка при обучении)"
    if xgboost_use_gpu else "CPU"
)


print("\n" + "=" * 60)
print("2. ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Не найден файл: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print("\nРазмер датасета:", df.shape)

print("\nПервые 5 строк:")
print(df.head())

print("\nИнформация о данных:")
df.info()

print("\nРаспределение Churn:")
print(df["Churn"].value_counts())

print("\nДоли классов:")
print(df["Churn"].value_counts(normalize=True))


print("\n" + "=" * 60)
print("3. ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 60)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nПропущенные значения:")
print(df.isnull().sum())

X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isnull().any():
    raise ValueError("Неизвестные значения Churn")

y = y.astype(int)

print("\nКоличество признаков:", X.shape[1])


print("\n" + "=" * 60)
print("4. РАЗДЕЛЕНИЕ ВЫБОРКИ")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Обучающая выборка:", X_train.shape[0])
print("Тестовая выборка:", X_test.shape[0])

print("\nРаспределение классов в обучающей выборке:")
print(y_train.value_counts())

print("\nРаспределение классов в тестовой выборке:")
print(y_test.value_counts())


print("\n" + "=" * 60)
print("5. ОБРАБОТКА ПРИЗНАКОВ")
print("=" * 60)

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

print("\nЧисловые признаки:")
print(numeric_features)

print("\nКатегориальные признаки:")
print(categorical_features)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


print("\n" + "=" * 60)
print("6. СОЗДАНИЕ МОДЕЛЕЙ")
print("=" * 60)


def create_catboost(use_gpu):
    params = {
        "iterations": 300,
        "depth": 5,
        "learning_rate": 0.05,
        "loss_function": "Logloss",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False
    }

    if use_gpu:
        params["task_type"] = "GPU"
        params["devices"] = "0"
    else:
        params["task_type"] = "CPU"
        params["thread_count"] = 2

    return CatBoostClassifier(**params)


def create_xgboost(use_gpu):
    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        n_jobs=2
    )


models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=15,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=1,
            random_state=42
        ),
        n_estimators=150,
        learning_rate=0.1,
        random_state=42
    ),

    "CatBoost": create_catboost(catboost_use_gpu),

    "XGBoost": create_xgboost(xgboost_use_gpu)
}


print("\n" + "=" * 60)
print("7. ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ")
print("=" * 60)

results = []
trained_models = {}

for model_name, model in models.items():

    print("\n" + "-" * 60)
    print("Обучение модели:", model_name)
    print("-" * 60)

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    if model_name == "CatBoost" and catboost_use_gpu:
        try:
            pipeline.fit(X_train, y_train)
            print("CatBoost обучен на GPU")

        except Exception as error:
            print("Ошибка CatBoost GPU:", error)
            print("Переключение CatBoost на CPU")

            pipeline = Pipeline(steps=[
                ("preprocessing", preprocessor),
                ("classifier", create_catboost(False))
            ])

            pipeline.fit(X_train, y_train)

    elif model_name == "XGBoost" and xgboost_use_gpu:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "error",
                    message=".*No visible GPU.*"
                )
                warnings.filterwarnings(
                    "error",
                    message=".*not compiled with CUDA support.*"
                )
                pipeline.fit(X_train, y_train)

            booster = pipeline.named_steps["classifier"].get_booster()
            actual_device = booster.attr("gpu_id")

            print("XGBoost: обучение завершено с запросом CUDA")
            print("Использование GPU зависит от сборки XGBoost")

        except Exception as error:
            print("Ошибка XGBoost GPU:", error)
            print("Переключение XGBoost на CPU")

            pipeline = Pipeline(steps=[
                ("preprocessing", preprocessor),
                ("classifier", create_xgboost(False))
            ])

            pipeline.fit(X_train, y_train)

    else:
        pipeline.fit(X_train, y_train)
        print("Обучение выполнено на CPU")

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    trained_models[model_name] = pipeline

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nОтчет классификации:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Нет оттока", "Отток"],
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Нет оттока", "Отток"],
        yticklabels=["Нет оттока", "Отток"]
    )

    plt.title(f"Матрица ошибок — {model_name}")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")

    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        RESULTS_DIR / f"confusion_{safe_name}.png",
        dpi=150
    )

    plt.close()


print("\n" + "=" * 60)
print("8. СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-score",
    ascending=False
).reset_index(drop=True)

print("\nИТОГОВАЯ ТАБЛИЦА:")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

results_df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 60)
print("9. ПОСТРОЕНИЕ ГРАФИКА F1-SCORE")
print("=" * 60)

plt.figure(figsize=(10, 6))

colors = sns.color_palette(
    "viridis",
    len(results_df)
)

bars = plt.bar(
    results_df["Model"],
    results_df["F1-score"],
    color=colors
)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,
        f"{height:.4f}",
        ha="center",
        fontsize=10
    )

plt.title(
    "Сравнение моделей по F1-score для класса Отток",
    fontsize=14
)

plt.xlabel("Модель")
plt.ylabel("F1-score")
plt.ylim(0, 1)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "f1_comparison.png",
    dpi=150
)

plt.show()


print("\n" + "=" * 60)
print("10. ОПРЕДЕЛЕНИЕ ЛУЧШЕЙ МОДЕЛИ")
print("=" * 60)

best_model_name = results_df.iloc[0]["Model"]
best_f1 = results_df.iloc[0]["F1-score"]

print("\nЛучшая модель:", best_model_name)
print(f"F1-score лучшей модели: {best_f1:.4f}")

print("\nВЫВОД:")

print(
    "В ходе лабораторной работы обучены пять моделей: "
    "Decision Tree, Random Forest, AdaBoost, CatBoost и XGBoost."
)

print(
    "Для оценки качества использовались Accuracy, "
    "Precision, Recall и F1-score."
)

print(
    f"Наибольший F1-score для класса 'Отток' "
    f"получила модель {best_model_name}: {best_f1:.4f}."
)

print(
    "Для удержания клиентов особенно важен Recall, "
    "поскольку он показывает долю обнаруженных клиентов, "
    "которые действительно ушли."
)

print(
    "Precision позволяет оценить, какая доля клиентов "
    "с прогнозом оттока действительно относится "
    "к группе уходящих."
)

print(
    "При выборе модели для практического применения "
    "нужно учитывать F1-score и стоимость ошибок."
)

print("\n" + "=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
print("Результаты сохранены в:", RESULTS_DIR)
print("=" * 60)
