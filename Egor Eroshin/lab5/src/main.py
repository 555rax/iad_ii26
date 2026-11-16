import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier  # Аналог CatBoost / LightGBM из sklearn
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("1. Загрузка датасета Wine Quality...")

try:
    df = pd.read_csv('winequality-white.csv', sep=';')
    print("-> Данные успешно загружены из локального файла winequality-white.csv")
except FileNotFoundError:
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/winequality-white.csv"
    df = pd.read_csv(url)
    print("-> Данные успешно скачаны по ссылке")

print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов.")


df['target'] = (df['quality'] >= 7).astype(int)

X = df.drop(columns=['quality', 'target'])
y = df['target']

print("\nРаспределение классов:")
print(f"Обычное вино (< 7):  {sum(y == 0)} образцов ({sum(y == 0) / len(y) * 100:.1f}%)")
print(f"Хорошее вино (>= 7): {sum(y == 1)} образцов ({sum(y == 1) / len(y) * 100:.1f}%)")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Single Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=RANDOM_STATE),
    "HistGB (CatBoost analog)": HistGradientBoostingClassifier(random_state=RANDOM_STATE)
}

results = []

print("\n2. Обучение моделей и вычисление метрик...")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

# Сортировка по F1-Score
results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
print("\n" + "=" * 65)
print(" СРАВНИТЕЛЬНЫЕ РЕЗУЛЬТАТЫ МОДЕЛЕЙ (Сортировка по F1-Score)")
print("=" * 65)
print(results_df.to_string(index=False))


plt.figure(figsize=(12, 5))
results_melted = pd.melt(results_df, id_vars=["Model"], value_vars=["Precision", "Recall", "F1-Score"],
                         var_name="Metric", value_name="Score")

sns.barplot(data=results_melted, x="Model", y="Score", hue="Metric", palette="Set2")
plt.title("Сравнение Precision, Recall и F1-Score (Вариант 3: Wine Quality)")
plt.ylim(0, 1.0)
plt.ylabel("Значение метрики")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

best_model_name = results_df.iloc[0]["Model"]
worst_model_name = results_df.iloc[-1]["Model"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, m_name, title in zip(axes, [worst_model_name, best_model_name], ["Худшая модель", "Лучшая модель"]):
    preds = models[m_name].predict(X_test_scaled)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Обычное (<7)', 'Хорошее (>=7)'],
                yticklabels=['Обычное (<7)', 'Хорошее (>=7)'])
    ax.set_title(f"{title}: {m_name}")
    ax.set_ylabel("Истинные классы")
    ax.set_xlabel("Предсказанные классы")

plt.suptitle("Сравнение Матриц Ошибок", fontsize=14)
plt.tight_layout()
plt.show()