import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix


# ==========================================
# 1. Загрузка данных
# ==========================================
def load_raisin_data():
    print("Загрузка датасета Raisin (Вариант 9)...")
    raisin = fetch_ucirepo(id=850)
    X = raisin.data.features.values
    y = raisin.data.targets.values.ravel()

    y = LabelEncoder().fit_transform(y)
    X = MinMaxScaler().fit_transform(X)  # Для RBM нужен масштаб [0, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    return train_loader, X_train_t.shape[1], X_train_t, X_test_t, y_test_t


def load_wine_data():
    print("Загрузка датасета Wine Quality (ЛР 2)...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, sep=';')

    X = df.drop(columns=["quality"]).values
    y = df["quality"].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(le.classes_)

    X = MinMaxScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
    return train_loader, X_train_t.shape[1], num_classes, X_train_t, X_test_t, y_test_t, le.classes_


# ==========================================
# 2. Архитектуры
# ==========================================
class ClassifierBinary(nn.Module):
    def __init__(self, input_dim):
        super(ClassifierBinary, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.classifier = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.sigmoid(self.classifier(x))


class ClassifierMulti(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassifierMulti, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.classifier = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.classifier(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        encoded = self.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded


class RBM():
    def __init__(self, visible_dim, hidden_dim, lr=0.01):
        self.W = torch.randn(hidden_dim, visible_dim) * 0.1
        self.h_bias = torch.zeros(hidden_dim)
        self.v_bias = torch.zeros(visible_dim)
        self.lr = lr

    def sample_h(self, v):
        activation = torch.mm(v, self.W.t()) + self.h_bias
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, h):
        activation = torch.mm(h, self.W) + self.v_bias
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train_step(self, v0):
        ph0, h0 = self.sample_h(v0)
        pv1, v1 = self.sample_v(h0)
        ph1, h1 = self.sample_h(v1)

        self.W += self.lr * (torch.mm(ph0.t(), v0) - torch.mm(ph1.t(), v1)) / v0.size(0)
        self.v_bias += self.lr * torch.sum(v0 - v1, dim=0) / v0.size(0)
        self.h_bias += self.lr * torch.sum(ph0 - ph1, dim=0) / v0.size(0)


# ==========================================
# 3. Функции обучения
# ==========================================
def train_classifier(model, train_loader, is_binary=True, epochs=50, lr=0.001):
    criterion = nn.BCELoss() if is_binary else nn.CrossEntropyLoss()
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


def pretrain_autoencoder(X_train, input_dim, epochs=40):
    ae1 = Autoencoder(input_dim, 64)
    optimizer1 = optim.Adam(ae1.parameters(), lr=0.005)
    loader1 = DataLoader(TensorDataset(X_train, X_train), batch_size=32, shuffle=True)
    for _ in range(epochs):
        for X_b, _ in loader1:
            optimizer1.zero_grad()
            loss = nn.MSELoss()(ae1(X_b)[1], X_b)
            loss.backward()
            optimizer1.step()

    with torch.no_grad():
        H1 = ae1(X_train)[0]

    ae2 = Autoencoder(64, 32)
    optimizer2 = optim.Adam(ae2.parameters(), lr=0.005)
    loader2 = DataLoader(TensorDataset(H1, H1), batch_size=32, shuffle=True)
    for _ in range(epochs):
        for X_b, _ in loader2:
            optimizer2.zero_grad()
            loss = nn.MSELoss()(ae2(X_b)[1], X_b)
            loss.backward()
            optimizer2.step()

    with torch.no_grad():
        H2 = ae2(H1)[0]

    ae3 = Autoencoder(32, 16)
    optimizer3 = optim.Adam(ae3.parameters(), lr=0.005)
    loader3 = DataLoader(TensorDataset(H2, H2), batch_size=32, shuffle=True)
    for _ in range(epochs):
        for X_b, _ in loader3:
            optimizer3.zero_grad()
            loss = nn.MSELoss()(ae3(X_b)[1], X_b)
            loss.backward()
            optimizer3.step()

    return ae1, ae2, ae3


def pretrain_rbm(X_train, input_dim, epochs=40):
    rbm1 = RBM(input_dim, 64)
    loader1 = DataLoader(TensorDataset(X_train), batch_size=32, shuffle=True)
    for _ in range(epochs):
        for X_b in loader1: rbm1.train_step(X_b[0])

    with torch.no_grad():
        H1, _ = rbm1.sample_h(X_train)

    rbm2 = RBM(64, 32)
    loader2 = DataLoader(TensorDataset(H1), batch_size=32, shuffle=True)
    for _ in range(epochs):
        for X_b in loader2: rbm2.train_step(X_b[0])

    with torch.no_grad():
        H2, _ = rbm2.sample_h(H1)

    rbm3 = RBM(32, 16)
    loader3 = DataLoader(TensorDataset(H2), batch_size=32, shuffle=True)
    for _ in range(epochs):
        for X_b in loader3: rbm3.train_step(X_b[0])

    return rbm1, rbm2, rbm3


# ==========================================
# 4. Основной пайплайн и Визуализация
# ==========================================
def evaluate_and_plot(models, losses, X_test, y_test, title_prefix, is_binary=True, class_names=None):
    y_true = y_test.numpy()
    preds = []

    for model in models:
        model.eval()
        with torch.no_grad():
            if is_binary:
                preds.append((model(X_test) >= 0.5).float().numpy())
            else:
                _, p = torch.max(model(X_test), 1)
                preds.append(p.numpy())

    avg_method = 'binary' if is_binary else 'weighted'
    f1_base = f1_score(y_true, preds[0], average=avg_method)
    f1_ae = f1_score(y_true, preds[1], average=avg_method)
    f1_rbm = f1_score(y_true, preds[2], average=avg_method)

    print(f"\n--- {title_prefix} ---")
    print(f"F1 (Base): {f1_base:.4f} | F1 (Autoencoder): {f1_ae:.4f} | F1 (RBM): {f1_rbm:.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(title_prefix, fontsize=16)

    axes[0].plot(losses[0], label='Base')
    axes[0].plot(losses[1], label='Autoencoder')
    axes[0].plot(losses[2], label='RBM')
    axes[0].set_title('Кривые обучения (Loss)')
    axes[0].legend()
    axes[0].grid(True)

    cm_kwargs = {'annot': True, 'fmt': 'd'}
    if class_names is not None:
        cm_kwargs['xticklabels'] = class_names
        cm_kwargs['yticklabels'] = class_names

    sns.heatmap(confusion_matrix(y_true, preds[0]), cmap='Blues', ax=axes[1], **cm_kwargs)
    axes[1].set_title(f'Без предобучения (F1: {f1_base:.3f})')

    sns.heatmap(confusion_matrix(y_true, preds[1]), cmap='Greens', ax=axes[2], **cm_kwargs)
    axes[2].set_title(f'Autoencoder (F1: {f1_ae:.3f})')

    sns.heatmap(confusion_matrix(y_true, preds[2]), cmap='Oranges', ax=axes[3], **cm_kwargs)
    axes[3].set_title(f'RBM (F1: {f1_rbm:.3f})')

    plt.tight_layout()
    plt.show()


def run_experiment(dataset_name):
    if dataset_name == "raisin":
        train_loader, input_dim, X_train, X_test, y_test = load_raisin_data()
        ModelClass = lambda: ClassifierBinary(input_dim)
        is_binary = True
        class_names = None
    else:
        train_loader, input_dim, num_classes, X_train, X_test, y_test, class_names = load_wine_data()
        ModelClass = lambda: ClassifierMulti(input_dim, num_classes)
        is_binary = False

    epochs_pre = 40
    epochs_fine = 60

    print(f"\n[1] Обучение БЕЗ предобучения ({dataset_name})...")
    model_base = ModelClass()
    loss_base = train_classifier(model_base, train_loader, is_binary, epochs_fine)

    print(f"[2] Обучение с Autoencoder ({dataset_name})...")
    ae1, ae2, ae3 = pretrain_autoencoder(X_train, input_dim, epochs_pre)
    model_ae = ModelClass()
    model_ae.layer1.weight.data, model_ae.layer1.bias.data = ae1.encoder.weight.data.clone(), ae1.encoder.bias.data.clone()
    model_ae.layer2.weight.data, model_ae.layer2.bias.data = ae2.encoder.weight.data.clone(), ae2.encoder.bias.data.clone()
    model_ae.layer3.weight.data, model_ae.layer3.bias.data = ae3.encoder.weight.data.clone(), ae3.encoder.bias.data.clone()
    loss_ae = train_classifier(model_ae, train_loader, is_binary, epochs_fine)

    print(f"[3] Обучение с RBM ({dataset_name})...")
    rbm1, rbm2, rbm3 = pretrain_rbm(X_train, input_dim, epochs_pre)
    model_rbm = ModelClass()
    model_rbm.layer1.weight.data, model_rbm.layer1.bias.data = rbm1.W.data.clone(), rbm1.h_bias.data.clone()
    model_rbm.layer2.weight.data, model_rbm.layer2.bias.data = rbm2.W.data.clone(), rbm2.h_bias.data.clone()
    model_rbm.layer3.weight.data, model_rbm.layer3.bias.data = rbm3.W.data.clone(), rbm3.h_bias.data.clone()
    loss_rbm = train_classifier(model_rbm, train_loader, is_binary, epochs_fine)

    evaluate_and_plot(
        [model_base, model_ae, model_rbm],
        [loss_base, loss_ae, loss_rbm],
        X_test, y_test,
        f"Результаты: {dataset_name.upper()}",
        is_binary,
        class_names
    )


if __name__ == "__main__":
    run_experiment("raisin")

    run_experiment("wine")