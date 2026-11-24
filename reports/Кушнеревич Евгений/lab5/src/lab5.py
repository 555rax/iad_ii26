import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


cols = ['area', 'perimeter', 'compactness', 'kernel_length',
        'kernel_width', 'asymmetry', 'groove_length', 'variety']

df = pd.read_csv('seeds_dataset.txt', sep=r'\s+', header=None, names=cols)

print("Размер:", df.shape)
print("\nПропуски:\n", df.isna().sum().sum())
print("\nКлассы:\n", df['variety'].value_counts().sort_index())
df.head()


class_names = ['Kama', 'Rosa', 'Canadian']

X = df.drop(columns='variety').values
y = df['variety'].values - 1          

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)   

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   
X_test_s  = scaler.transform(X_test)

print("train:", X_train_s.shape, " test:", X_test_s.shape)


models = {
    'Одиночное дерево': DecisionTreeClassifier(max_depth=4, random_state=42),
    'Случайный лес': RandomForestClassifier(n_estimators=200, random_state=42),
    'AdaBoost': AdaBoostClassifier(
                            estimator=DecisionTreeClassifier(max_depth=1),  
                            n_estimators=200, learning_rate=0.5, random_state=42),
    'XGBoost':  XGBClassifier(n_estimators=200, max_depth=3,
                                      learning_rate=0.1, eval_metric='mlogloss',
                                      random_state=42),
    'CatBoost': CatBoostClassifier(iterations=300, depth=4,
                                           learning_rate=0.1, verbose=0,
                                           random_seed=42),
}

preds = {}
rows = []
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred_test = np.asarray(model.predict(X_test_s)).ravel()
    pred_train = np.asarray(model.predict(X_train_s)).ravel()
    preds[name] = pred_test
    rows.append({
        'Модель': name,
        'Train accuracy': accuracy_score(y_train, pred_train),
        'Test accuracy':  accuracy_score(y_test, pred_test),
    })

results = pd.DataFrame(rows).set_index('Модель')
results.round(4)


ax = results['Test accuracy'].plot(kind='bar', figsize=(8, 4), color='steelblue', rot=15)
ax.set_ylim(results['Test accuracy'].min() - 0.05, 1.0)
ax.set_ylabel('Accuracy на тесте')
ax.set_title('Сравнение моделей на Seeds')
for i, v in enumerate(results['Test accuracy']):
    ax.text(i, v + 0.003, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.show()


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_rows = []
for name, model in models.items():
    pipe = make_pipeline(StandardScaler(), clone(model))
    scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
    cv_rows.append({'Модель': name,
                    'CV accuracy (среднее)': scores.mean(),
                    'CV (std)': scores.std()})

cv_results = pd.DataFrame(cv_rows).set_index('Модель')
cv_results.round(4)


best_name = results['Test accuracy'].idxmax()
y_pred_best = preds[best_name]
print("Лучшая модель на тесте:", best_name)
print()
print(classification_report(y_test, y_pred_best, target_names=class_names))

cm = confusion_matrix(y_test, y_pred_best)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap='Blues')
plt.title(f'Матрица ошибок: {best_name}')
plt.show()


pca = PCA(n_components=2)
pca.fit(X_train_s)
X_test_2d = pca.transform(X_test_s)
print("Доля дисперсии, объяснённая 2 компонентами:",
      pca.explained_variance_ratio_.sum().round(3))

colors = ['tab:blue', 'tab:orange', 'tab:green']
errors = y_pred_best != y_test

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

for k, cname in enumerate(class_names):
    m = y_test == k
    axes[0].scatter(X_test_2d[m, 0], X_test_2d[m, 1], c=colors[k],
                    label=cname, s=60, edgecolor='k')
axes[0].set_title('Истинные сорта (тест)')

for k, cname in enumerate(class_names):
    m = y_pred_best == k
    axes[1].scatter(X_test_2d[m, 0], X_test_2d[m, 1], c=colors[k],
                    label=cname, s=60, edgecolor='k')
axes[1].scatter(X_test_2d[errors, 0], X_test_2d[errors, 1],
                facecolors='none', edgecolors='red', s=250, linewidths=2,
                label='Ошибка')
axes[1].set_title(f'Предсказания: {best_name}')

for ax in axes:
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()