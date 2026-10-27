import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix


def load_and_prepare_wine_data():
    print("Загрузка датасета Wine Quality (White)...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, sep=';')

    X = df.drop(columns=["quality"]).values
    y = df["quality"].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(le.classes_)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    return train_loader, X_train_t.shape[1], num_classes, X_train_t, X_test_t, y_test_t, le.classes_
class DeepWineClassifier(nn.Module):
    """Полносвязная архитектура (4 слоя) для многоклассовой классификации"""

    def __init__(self, input_dim, num_classes):
        super(DeepWineClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.classifier = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.classifier(x)  # Без Softmax (используем CrossEntropyLoss)


class Autoencoder(nn.Module):
    """Автоэнкодер для предобучения одного слоя"""

    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        encoded = self.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_classifier(model, train_loader, epochs=70, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        loss_history.append(epoch_loss / len(train_loader.dataset))
    return loss_history


def train_autoencoder(ae, data_tensor, epochs=50, lr=0.005):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    ae.train()
    dataset = TensorDataset(data_tensor, data_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        for X_batch, _ in loader:
            optimizer.zero_grad()
            _, decoded = ae(X_batch)
            loss = criterion(decoded, X_batch)
            loss.backward()
            optimizer.step()
    return ae


def evaluate_and_plot(model_base, model_pre, X_test, y_test, loss_base, loss_pre, class_names):
    model_base.eval()
    model_pre.eval()

    with torch.no_grad():
        _, preds_base = torch.max(model_base(X_test), 1)
        _, preds_pre = torch.max(model_pre(X_test), 1)

    y_true = y_test.numpy()
    preds_base = preds_base.numpy()
    preds_pre = preds_pre.numpy()

    f1_base = f1_score(y_true, preds_base, average='weighted')
    f1_pre = f1_score(y_true, preds_pre, average='weighted')

    cm_base = confusion_matrix(y_true, preds_base)
    cm_pre = confusion_matrix(y_true, preds_pre)

    print(f"\n--- Результаты (ЛР 2: Wine Quality) ---")
    print(f"F1 Score (Weighted) БЕЗ предобучения: {f1_base:.4f}")
    print(f"F1 Score (Weighted) С предобучением: {f1_pre:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].plot(loss_base, label='БЕЗ предобучения', color='red')
    axes[0].plot(loss_pre, label='С предобучением', color='green')
    axes[0].set_title('Кривые обучения (CrossEntropy Loss)')
    axes[0].set_xlabel('Эпохи')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[1], xticklabels=class_names,
                yticklabels=class_names)
    axes[1].set_title(f'Без предобучения (F1: {f1_base:.3f})')
    axes[1].set_xlabel('Предсказано')
    axes[1].set_ylabel('Истина')

    sns.heatmap(cm_pre, annot=True, fmt='d', cmap='Greens', ax=axes[2], xticklabels=class_names,
                yticklabels=class_names)
    axes[2].set_title(f'С предобучением (F1: {f1_pre:.3f})')
    axes[2].set_xlabel('Предсказано')
    axes[2].set_ylabel('Истина')

    plt.tight_layout()
    plt.show()


# ==========================================
# 4. Основной пайплайн
# ==========================================
def main():
    train_loader, input_dim, num_classes, X_train, X_test, y_test, class_names = load_and_prepare_wine_data()

    print("\n[1] Обучение модели БЕЗ предобучения...")
    baseline_model = DeepWineClassifier(input_dim, num_classes)
    loss_history_base = train_classifier(baseline_model, train_loader, epochs=70)

    print("\n[2] Запуск послойного автоэнкодерного предобучения...")
    ae1 = Autoencoder(input_dim, 64)
    train_autoencoder(ae1, X_train, epochs=50)
    with torch.no_grad(): H1, _ = ae1(X_train)

    ae2 = Autoencoder(64, 32)
    train_autoencoder(ae2, H1, epochs=50)
    with torch.no_grad(): H2, _ = ae2(H1)

    ae3 = Autoencoder(32, 16)
    train_autoencoder(ae3, H2, epochs=50)

    print("\n[3] Перенос весов и дообучение...")
    pretrained_model = DeepWineClassifier(input_dim, num_classes)

    pretrained_model.layer1.weight.data = ae1.encoder.weight.data.clone()
    pretrained_model.layer1.bias.data = ae1.encoder.bias.data.clone()
    pretrained_model.layer2.weight.data = ae2.encoder.weight.data.clone()
    pretrained_model.layer2.bias.data = ae2.encoder.bias.data.clone()
    pretrained_model.layer3.weight.data = ae3.encoder.weight.data.clone()
    pretrained_model.layer3.bias.data = ae3.encoder.bias.data.clone()

    loss_history_pre = train_classifier(pretrained_model, train_loader, epochs=70)

    evaluate_and_plot(baseline_model, pretrained_model, X_test, y_test, loss_history_base, loss_history_pre,
                      class_names)


if __name__ == "__main__":
    main()