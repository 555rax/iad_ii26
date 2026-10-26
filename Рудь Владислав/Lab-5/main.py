import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# Задание 1. Загрузка данных и ознакомление
# ==========================================
print("--- Задание 1: Загрузка данных ---")
df = pd.read_csv('iris.csv')
print("Первые 5 строк датасета:")
print(df.head())
print("\nИнформация о датасете:")
print(df.info())

# Подготовка данных
X = df.drop('variety', axis=1)
y = df['variety']

# Кодирование целевой переменной (Setosa, Versicolor, Virginica -> 0, 1, 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ==========================================
# Задание 2. Разделение данных
# ==========================================
print("\n--- Задание 2: Разделение данных ---")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(f"Размер обучающей выборки: {X_train.shape[0]} объектов")
print(f"Размер тестовой выборки: {X_test.shape[0]} объектов")

# ==========================================
# Задание 3. Обучение моделей
# ==========================================
print("\n--- Задание 3: Обучение моделей ---")

# Инициализация моделей
models = {
    "Одиночное дерево": DecisionTreeClassifier(random_state=42),
    "Случайный лес": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss'),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"Модель {name} успешно обучена.")

# ==========================================
# Задание 4. Оценка точности
# ==========================================
print("\n--- Задание 4: Оценка точности (Accuracy) ---")
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name}: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# ==========================================
# Дополнительно: Визуализация (Графики)
# ==========================================
print("\n--- Построение графиков ---")

# Настройка стиля графиков
sns.set_theme(style="whitegrid")

# ГРАФИК 1: Сравнение точности моделей
plt.figure(figsize=(10, 5))
names = list(results.keys())
accuracies = list(results.values())

bars = plt.barh(names, accuracies, color='skyblue')
plt.xlabel('Точность (Accuracy)')
plt.title('Сравнение точности моделей на тестовой выборке')
plt.xlim(0, 1.1)

# Добавляем проценты на столбцы
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f'{width*100:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('accuracy_comparison.png') # Сохраняем картинку для отчета
plt.show()

# ГРАФИК 2: Матрица ошибок для AdaBoost
ada_model = models["AdaBoost"]
y_pred_ada = ada_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_ada)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Матрица ошибок (AdaBoost)')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# ГРАФИК 3: Важность признаков (Random Forest)
rf_model = models["Случайный лес"]
importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=feature_names, hue=feature_names, palette='viridis', legend=False)
plt.title('Важность признаков (Случайный лес)')
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\nВсе графики построены и сохранены в папку с проектом!")