import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

df = pd.read_csv('bank.csv', sep=None, engine='python')
df.columns = [c.strip().strip('"') for c in df.columns]

target_col = 'y' if 'y' in df.columns else 'deposit'
y = df[target_col].astype(str).str.strip().str.strip('"').str.lower().map({'yes': 1, 'no': 0}).values
X = df.drop(columns=[target_col])

print("Размер данных:", X.shape)
print("Доля класса 'yes':", y.mean())

cat_cols = X.select_dtypes(include=['object', 'str']).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

neg, pos = np.sum(y_train == 0), np.sum(y_train == 1)

models = {
    'Решающее дерево': DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
    'Случайный лес': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1),
    'AdaBoost': AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                             scale_pos_weight=neg / pos, eval_metric='logloss', random_state=42),
    'CatBoost': CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                                   auto_class_weights='Balanced', verbose=0, random_seed=42),
}

results = {}
cms = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        'F1': f1_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
    }
    cms[name] = confusion_matrix(y_test, preds)
    print(f"\n{name}")
    print(f"F1 = {results[name]['F1']:.4f}, Precision = {results[name]['Precision']:.4f}, Recall = {results[name]['Recall']:.4f}")
    print("Confusion matrix:\n", cms[name])

res_df = pd.DataFrame(results).T.sort_values('F1', ascending=False)
print("\nИтоговая таблица (по убыванию F1):")
print(res_df.round(4))

best_f1 = res_df['F1'].idxmax()
best_recall = res_df['Recall'].idxmax()
print(f"\nЛучший F1: {best_f1} ({res_df.loc[best_f1, 'F1']:.4f})")
print(f"Лучший Recall: {best_recall} ({res_df.loc[best_recall, 'Recall']:.4f})")

plt.figure(figsize=(10, 6))
res_df.plot(kind='bar', ax=plt.gca())
plt.ylabel('Значение метрики')
plt.title('Сравнение моделей (класс "yes")')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('lab5_metrics.png')
plt.show()

fig, axes = plt.subplots(1, len(cms), figsize=(4.5 * len(cms), 4))
for ax, (name, cm) in zip(axes, cms.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(name)
    ax.set_xlabel('Предсказано')
    ax.set_ylabel('Истина')
plt.tight_layout()
plt.savefig('lab5_confusion_matrices.png')
plt.show()