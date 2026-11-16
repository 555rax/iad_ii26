import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')


torch.manual_seed(42)
np.random.seed(42)

class SingleAutoencoder(nn.Module):
    """Одиночный автоэнкодер для послойного обучения"""

    def __init__(self, in_features, out_features):
        super(SingleAutoencoder, self).__init__()
        self.encoder = nn.Linear(in_features, out_features)
        self.decoder = nn.Linear(out_features, in_features)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.encoder(x))
        rec = self.decoder(h)
        return rec, h


class RBM(nn.Module):
    """Ограниченная машина Больцмана (Restricted Boltzmann Machine)"""

    def __init__(self, visible_dim, hidden_dim, lr=0.01):
        super(RBM, self).__init__()
        # Инициализация весов и смещений
        self.W = nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.v_bias = nn.Parameter(torch.zeros(visible_dim))
        self.lr = lr

    def sample_hidden(self, v):

        h_prob = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)

        h_sample = torch.bernoulli(h_prob)
        return h_prob, h_sample

    def sample_visible(self, h):

        v_prob = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        v_sample = torch.bernoulli(v_prob)
        return v_prob, v_sample

    def forward(self, v):

        h_prob_0, h_sample_0 = self.sample_hidden(v)


        v_prob_1, _ = self.sample_visible(h_sample_0)
        h_prob_1, _ = self.sample_hidden(v_prob_1)


        pos_assoc = torch.matmul(v.t(), h_prob_0)
        neg_assoc = torch.matmul(v_prob_1.t(), h_prob_1)

        batch_size = v.size(0)
        self.W.data += self.lr * (pos_assoc - neg_assoc) / batch_size
        self.v_bias.data += self.lr * torch.mean(v - v_prob_1, dim=0)
        self.h_bias.data += self.lr * torch.mean(h_prob_0 - h_prob_1, dim=0)

        return h_prob_0


class DeepClassifier(nn.Module):
    """Глубокий классификатор, который будем обучать 3-мя разными способами"""

    def __init__(self, input_dim, num_classes):
        super(DeepClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.out(x)


def pretrain_ae_layers(X_train, input_dim, epochs=50, lr=0.01):
    print('  -> Запуск послойного автоэнкодерного (AE) предобучения...')

    def train_ae(model, data, epochs, lr):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        tensor_data = torch.FloatTensor(data)
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed, _ = model(tensor_data)
            loss = criterion(reconstructed, tensor_data)
            loss.backward()
            optimizer.step()

    ae1 = SingleAutoencoder(input_dim, 64)
    train_ae(ae1, X_train, epochs, lr)
    _, H1 = ae1(torch.FloatTensor(X_train))

    ae2 = SingleAutoencoder(64, 32)
    train_ae(ae2, H1.detach().numpy(), epochs, lr)
    _, H2 = ae2(H1)

    ae3 = SingleAutoencoder(32, 16)
    train_ae(ae3, H2.detach().numpy(), epochs, lr)

    return ae1, ae2, ae3


def pretrain_rbm_layers(X_train, input_dim, epochs=50, lr=0.05):
    print('  -> Запуск послойного RBM предобучения...')

    # RBM работает с вероятностями [0, 1]. Применяем MinMax поверх Standard Scaler
    scaler = MinMaxScaler()
    X_train_minmax = scaler.fit_transform(X_train)
    tensor_data = torch.FloatTensor(X_train_minmax)

    def train_rbm(rbm, data, epochs):
        for epoch in range(epochs):
            h_out = rbm(data)
        return h_out

    rbm1 = RBM(input_dim, 64, lr)
    H1 = train_rbm(rbm1, tensor_data, epochs)

    rbm2 = RBM(64, 32, lr)
    H2 = train_rbm(rbm2, H1.detach(), epochs)

    rbm3 = RBM(32, 16, lr)
    train_rbm(rbm3, H2.detach(), epochs)

    return rbm1, rbm2, rbm3

def run_experiment(X, y, dataset_name):
    print(f'\n{"=" * 60}')
    print(f' ЭКСПЕРИМЕНТ: {dataset_name}')
    print(f'{"=" * 60}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_train_scaled.shape[1]
    num_classes = len(np.unique(y))

    X_tr_t = torch.FloatTensor(X_train_scaled)
    y_tr_t = torch.LongTensor(y_train)
    X_te_t = torch.FloatTensor(X_test_scaled)

    criterion = nn.CrossEntropyLoss()
    fine_tune_epochs = 100


    model_std = DeepClassifier(input_dim, num_classes)
    optimizer_std = optim.Adam(model_std.parameters(), lr=0.005)

    for epoch in range(fine_tune_epochs):
        optimizer_std.zero_grad()
        loss = criterion(model_std(X_tr_t), y_tr_t)
        loss.backward()
        optimizer_std.step()

    with torch.no_grad():
        preds_std = torch.argmax(model_std(X_te_t), dim=1).numpy()

    f1_std = f1_score(y_test, preds_std, average='weighted')
    acc_std = accuracy_score(y_test, preds_std)


    ae1, ae2, ae3 = pretrain_ae_layers(X_train_scaled, input_dim)
    model_ae = DeepClassifier(input_dim, num_classes)

    with torch.no_grad():
        model_ae.fc1.weight.copy_(ae1.encoder.weight)
        model_ae.fc1.bias.copy_(ae1.encoder.bias)
        model_ae.fc2.weight.copy_(ae2.encoder.weight)
        model_ae.fc2.bias.copy_(ae2.encoder.bias)
        model_ae.fc3.weight.copy_(ae3.encoder.weight)
        model_ae.fc3.bias.copy_(ae3.encoder.bias)

    optimizer_ae = optim.Adam(model_ae.parameters(), lr=0.002)
    for epoch in range(fine_tune_epochs):
        optimizer_ae.zero_grad()
        loss = criterion(model_ae(X_tr_t), y_tr_t)
        loss.backward()
        optimizer_ae.step()

    with torch.no_grad():
        preds_ae = torch.argmax(model_ae(X_te_t), dim=1).numpy()

    f1_ae = f1_score(y_test, preds_ae, average='weighted')
    acc_ae = accuracy_score(y_test, preds_ae)


    rbm1, rbm2, rbm3 = pretrain_rbm_layers(X_train_scaled, input_dim)
    model_rbm = DeepClassifier(input_dim, num_classes)

    with torch.no_grad():
        model_rbm.fc1.weight.copy_(rbm1.W.t())
        model_rbm.fc1.bias.copy_(rbm1.h_bias)
        model_rbm.fc2.weight.copy_(rbm2.W.t())
        model_rbm.fc2.bias.copy_(rbm2.h_bias)
        model_rbm.fc3.weight.copy_(rbm3.W.t())
        model_rbm.fc3.bias.copy_(rbm3.h_bias)

    optimizer_rbm = optim.Adam(model_rbm.parameters(), lr=0.002)
    for epoch in range(fine_tune_epochs):
        optimizer_rbm.zero_grad()
        loss = criterion(model_rbm(X_tr_t), y_tr_t)
        loss.backward()
        optimizer_rbm.step()

    with torch.no_grad():
        preds_rbm = torch.argmax(model_rbm(X_te_t), dim=1).numpy()

    f1_rbm = f1_score(y_test, preds_rbm, average='weighted')
    acc_rbm = accuracy_score(y_test, preds_rbm)

    print(f'\nИТОГОВОЕ СРАВНЕНИЕ ({dataset_name}):')
    print(f'1. Без предобучения : Accuracy = {acc_std:.4f} | F1-Score = {f1_std:.4f}')
    print(f'2. Предобучение АЭ  : Accuracy = {acc_ae:.4f}  | F1-Score = {f1_ae:.4f}')
    print(f'3. Предобучение RBM : Accuracy = {acc_rbm:.4f} | F1-Score = {f1_rbm:.4f}')

    # Визуализация Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(confusion_matrix(y_test, preds_std), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Без предобучения\n(F1: {f1_std:.4f})')

    sns.heatmap(confusion_matrix(y_test, preds_ae), annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title(f'Предобучение Автоэнкодером\n(F1: {f1_ae:.4f})')

    sns.heatmap(confusion_matrix(y_test, preds_rbm), annot=True, fmt='d', cmap='Reds', ax=axes[2])
    axes[2].set_title(f'Предобучение RBM\n(F1: {f1_rbm:.4f})')

    plt.suptitle(f'Сравнение матриц ошибок: {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.show()

print('Загрузка датасета 1: Maternal Health Risk (Вариант 3)...')
try:
    from ucimlrepo import fetch_ucirepo

    maternal = fetch_ucirepo(id=863)
    X1 = maternal.data.features.values
    y1_raw = maternal.data.targets['RiskLevel'].values
except:
    df1 = pd.read_csv(
        'https://raw.githubusercontent.com/datasets/maternal-health-risk/main/Maternal%20Health%20Risk%20Data%20Set.csv')
    X1 = df1.iloc[:, :-1].values
    y1_raw = df1.iloc[:, -1].values

y1 = LabelEncoder().fit_transform(y1_raw)
run_experiment(X1, y1, 'Maternal Health Risk (Вариант 3)')

print('\nЗагрузка датасета 2: Rice Dataset (Данные из ЛР 2)...')
try:
    from ucimlrepo import fetch_ucirepo

    rice = fetch_ucirepo(id=545)
    X2 = rice.data.features.values
    y2_raw = rice.data.targets['Class'].values
except:
    df2 = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/rice_cammeo_osmancik.csv')
    X2 = df2.iloc[:, :-1].values
    y2_raw = df2.iloc[:, -1].values

y2 = LabelEncoder().fit_transform(y2_raw)
run_experiment(X2, y2, 'Rice Dataset (Из ЛР 2)')