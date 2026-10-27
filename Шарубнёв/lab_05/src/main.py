import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')


def load_and_prepare_data():
    print("Загрузка датасета Glass Identification...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
    df = pd.read_csv(url, names=columns)

    df = df.drop('Id', axis=1)

    X = df.drop('Type', axis=1)
    y = df['Type']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test, le, X.columns


def plot_metrics_comparison(metrics_dict):
    names = list(metrics_dict.keys())
    accuracies = [metrics_dict[name]['accuracy'] for name in names]
    f1_scores = [metrics_dict[name]['f1_macro'] for name in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, accuracies, width, label='Accuracy', color='skyblue')
    ax.bar(x + width / 2, f1_scores, width, label='Macro F1-Score', color='salmon')

    ax.set_ylabel('Значение метрики')
    ax.set_title('Сравнение моделей по Accuracy и Macro F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(metrics_dict, class_names):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(metrics_dict.items()):
        cm = data['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=class_names, yticklabels=class_names)
        axes[idx].set_title(f'{name}')
        axes[idx].set_xlabel('Предсказано')
        axes[idx].set_ylabel('Истина')

    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.show()


def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.title("Важность признаков (Random Forest)")
    plt.bar(range(len(importances)), importances[indices], color="mediumseagreen", align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[indices])
    plt.xlim([-1, len(importances)])
    plt.ylabel('Относительная важность')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    X_train, X_test, y_train, y_test, le, feature_names = load_and_prepare_data()

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }

    print("\n--- Результаты моделей ---\n")

    target_names = [f"Type {cls}" for cls in le.classes_]
    metrics_dict = {}

    for name, model in models.items():
        print(f"Обучение {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics_dict[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        print(f"--- Classification Report: {name} ---")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        print("-" * 50)

    print("\nПостроение графиков...")
    plot_metrics_comparison(metrics_dict)
    plot_confusion_matrices(metrics_dict, target_names)
    plot_feature_importances(models["Random Forest"], feature_names)


if __name__ == "__main__":
    main()