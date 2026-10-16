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
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)


class SingleAutoencoder(nn.Module):

  def __init__(self, in_features, out_features):
    super(SingleAutoencoder, self).__init__()
    self.encoder = nn.Linear(in_features, out_features)
    self.decoder = nn.Linear(out_features, in_features)
    self.act = nn.ReLU()

  def forward(self, x):
    h = self.act(self.encoder(x))
    rec = self.decoder(h)
    return rec, h

class DeepClassifier(nn.Module):

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


def pretrain_layers(X_train, input_dim, epochs=50, lr=0.01):
  print('Запуск послойного автоэнкодерного предобучения')

  # Слой 1: input_dim -> 64
  ae1 = SingleAutoencoder(input_dim, 64)
  train_ae(ae1, X_train, epochs, lr)
  _, H1 = ae1(torch.FloatTensor(X_train))

  # Слой 2: 64 -> 32
  ae2 = SingleAutoencoder(64, 32)
  train_ae(ae2, H1.detach().numpy(), epochs, lr)
  _, H2 = ae2(H1)

  # Слой 3: 32 -> 16
  ae3 = SingleAutoencoder(32, 16)
  train_ae(ae3, H2.detach().numpy(), epochs, lr)

  return ae1, ae2, ae3


def train_ae(model, data, epochs, lr):
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.MSELoss()
  tensor_data = torch.FloatTensor(data)

  model.train()
  for epoch in range(epochs):
    optimizer.zero_grad()
    reconstructed, _ = model(tensor_data)
    loss = criterion(reconstructed, tensor_data)
    loss.backward()
    optimizer.step()


def run_experiment(X, y, dataset_name):
  print(f'\n')
  print(f'   ЭКСПЕРИМЕНТ: {dataset_name}')
  print(f'')

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

  model_std = DeepClassifier(input_dim, num_classes)
  optimizer_std = optim.Adam(model_std.parameters(), lr=0.005)
  criterion = nn.CrossEntropyLoss()

  model_std.train()
  for epoch in range(100):
    optimizer_std.zero_grad()
    out = model_std(X_tr_t)
    loss = criterion(out, y_tr_t)
    loss.backward()
    optimizer_std.step()

  model_std.eval()
  with torch.no_grad():
    preds_std = torch.argmax(model_std(X_te_t), dim=1).numpy()

  f1_std = f1_score(y_test, preds_std, average='weighted')
  acc_std = accuracy_score(y_test, preds_std)


  ae1, ae2, ae3 = pretrain_layers(X_train_scaled, input_dim)

  model_pre = DeepClassifier(input_dim, num_classes)


  with torch.no_grad():
    model_pre.fc1.weight.copy_(ae1.encoder.weight)
    model_pre.fc1.bias.copy_(ae1.encoder.bias)
    model_pre.fc2.weight.copy_(ae2.encoder.weight)
    model_pre.fc2.bias.copy_(ae2.encoder.bias)
    model_pre.fc3.weight.copy_(ae3.encoder.weight)
    model_pre.fc3.bias.copy_(ae3.encoder.bias)


  optimizer_pre = optim.Adam(
      model_pre.parameters(), lr=0.002
  )  # Чуть меньший LR для fine-tuning

  model_pre.train()
  for epoch in range(100):
    optimizer_pre.zero_grad()
    out = model_pre(X_tr_t)
    loss = criterion(out, y_tr_t)
    loss.backward()
    optimizer_pre.step()

  model_pre.eval()
  with torch.no_grad():
    preds_pre = torch.argmax(model_pre(X_te_t), dim=1).numpy()

  f1_pre = f1_score(y_test, preds_pre, average='weighted')
  acc_pre = accuracy_score(y_test, preds_pre)

  print(f'\nСравнение результатов ({dataset_name}) ---')
  print(f'Без предобучения   -> Accuracy: {acc_std:.4f} | F1-Score: {f1_std:.4f}')
  print(f'С предобучением    -> Accuracy: {acc_pre:.4f} | F1-Score: {f1_pre:.4f}')

  # Визуализация Confusion Matrices
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  sns.heatmap(
      confusion_matrix(y_test, preds_std),
      annot=True,
      fmt='d',
      cmap='Blues',
      ax=axes[0],
  )
  axes[0].set_title(f'Без предобучения\n(F1: {f1_std:.4f})')

  sns.heatmap(
      confusion_matrix(y_test, preds_pre),
      annot=True,
      fmt='d',
      cmap='Greens',
      ax=axes[1],
  )
  axes[1].set_title(f'С автоэнкодерным предобучением\n(F1: {f1_pre:.4f})')

  plt.suptitle(f'Confusion Matrix Comparison: {dataset_name}')
  plt.tight_layout()
  plt.show()


print('Загрузка датасета 1: Maternal Health Risk...')
url_maternal = 'https://archive.ics.uci.edu/static/public/863/maternal+health+risk.zip'
try:
  from ucimlrepo import fetch_ucirepo

  maternal = fetch_ucirepo(id=863)
  X1 = maternal.data.features.values
  y1_raw = maternal.data.targets['RiskLevel'].values
except:

  df1 = pd.read_csv(
      'https://raw.githubusercontent.com/datasets/maternal-health-risk/main/Maternal%20Health%20Risk%20Data%20Set.csv'
  )
  X1 = df1.iloc[:, :-1].values
  y1_raw = df1.iloc[:, -1].values

y1 = LabelEncoder().fit_transform(y1_raw)
run_experiment(X1, y1, 'Maternal Health Risk (Вариант 3)')


print('Загрузка датасета 2: Rice Dataset...')
try:
  from ucimlrepo import fetch_ucirepo

  rice = fetch_ucirepo(id=545)
  X2 = rice.data.features.values
  y2_raw = rice.data.targets['Class'].values
except:
  df2 = pd.read_csv(
      'https://raw.githubusercontent.com/jbrownlee/Datasets/master/rice_cammeo_osmancik.csv'
  )
  X2 = df2.iloc[:, :-1].values
  y2_raw = df2.iloc[:, -1].values

y2 = LabelEncoder().fit_transform(y2_raw)
run_experiment(X2, y2, 'Rice Dataset (Из ЛР 2)')