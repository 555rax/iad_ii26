import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

print("1. Загрузка и обработка данных...")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
df = pd.read_csv(url, header=None)

df = df.replace('?', np.nan)
df = df.fillna(df.mode().iloc[0])
df[15] = df[15].map({'+': 1, '-': 0})

X = df.drop(15, axis=1)
y = df[15]

X = pd.get_dummies(X)
X.columns = X.columns.astype(str)
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("2. Построение графиков данных...")
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
fig1.suptitle('Data & Autoencoder', fontsize=16)
y.value_counts().plot(kind='bar', ax=axes1[0], color=['skyblue', 'salmon'], edgecolor='k')
axes1[0].set_title('Class Distribution')
axes1[0].set_xticklabels(['Negative (0)', 'Positive (1)'], rotation=0)
axes1[0].set_ylabel('Count')

print("3. Обучение базовой модели (смотри прогресс-бар)...")
def build_base_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

base_model = build_base_model(X_train.shape[1])
history_base = base_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
y_prob_base = base_model.predict(X_test).ravel()
y_pred_base = (y_prob_base > 0.5).astype(int)
f1_base = f1_score(y_test, y_pred_base)
cm_base = confusion_matrix(y_test, y_pred_base)

print("4. Обучение автоэнкодера (смотри прогресс-бар)...")
def build_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(inputs)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    outputs = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(inputs=inputs, outputs=outputs)
    encoder = Model(inputs=inputs, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder(X_train.shape[1])
history_ae = autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

axes1[1].plot(history_ae.history['loss'], label='Train Loss', color='purple')
axes1[1].plot(history_ae.history['val_loss'], label='Val Loss', color='orange')
axes1[1].set_title('Autoencoder Reconstruction Loss')
axes1[1].set_xlabel('Epoch')
axes1[1].set_ylabel('MSE')
axes1[1].legend()
axes1[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('1_ae_loss.png')  # Сохраняем картинку
plt.show()

for layer in encoder.layers:
    layer.trainable = False

print("5. Обучение модели с предобучением (смотри прогресс-бар)...")
pretrained_input = Input(shape=(X_train.shape[1],))
pretrained_x = encoder(pretrained_input)
pretrained_x = Dense(16, activation='relu')(pretrained_x)
pretrained_output = Dense(1, activation='sigmoid')(pretrained_x)
pretrained_model = Model(inputs=pretrained_input, outputs=pretrained_output)
pretrained_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_pre = pretrained_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

y_prob_pre = pretrained_model.predict(X_test).ravel()
y_pred_pre = (y_prob_pre > 0.5).astype(int)
f1_pre = f1_score(y_test, y_pred_pre)
cm_pre = confusion_matrix(y_test, y_pred_pre)

print("6. Отрисовка финальных графиков...")
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
fig2.suptitle('Training History', fontsize=16)

axes2[0, 0].plot(history_base.history['loss'], label='Train', color='blue')
axes2[0, 0].plot(history_base.history['val_loss'], label='Val', color='red')
axes2[0, 0].set_title('Base Model Loss')
axes2[0, 0].set_ylabel('Loss')
axes2[0, 0].legend()
axes2[0, 0].grid(True, alpha=0.3)

axes2[0, 1].plot(history_base.history['accuracy'], label='Train', color='blue')
axes2[0, 1].plot(history_base.history['val_accuracy'], label='Val', color='red')
axes2[0, 1].set_title('Base Model Accuracy')
axes2[0, 1].set_ylabel('Accuracy')
axes2[0, 1].legend()
axes2[0, 1].grid(True, alpha=0.3)

axes2[1, 0].plot(history_pre.history['loss'], label='Train', color='green')
axes2[1, 0].plot(history_pre.history['val_loss'], label='Val', color='orange')
axes2[1, 0].set_title('Pretrained Model Loss')
axes2[1, 0].set_xlabel('Epoch')
axes2[1, 0].set_ylabel('Loss')
axes2[1, 0].legend()
axes2[1, 0].grid(True, alpha=0.3)

axes2[1, 1].plot(history_pre.history['accuracy'], label='Train', color='green')
axes2[1, 1].plot(history_pre.history['val_accuracy'], label='Val', color='orange')
axes2[1, 1].set_title('Pretrained Model Accuracy')
axes2[1, 1].set_xlabel('Epoch')
axes2[1, 1].set_ylabel('Accuracy')
axes2[1, 1].legend()
axes2[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('2_training_history.png')  # Сохраняем картинку
plt.show()

fig3 = plt.figure(figsize=(18, 5))
fig3.suptitle('Model Comparison', fontsize=16)

ax1 = fig3.add_subplot(131)
im1 = ax1.imshow(cm_base, interpolation='nearest', cmap=plt.cm.Blues)
ax1.set_title('Base Model CM')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
for i in range(2):
    for j in range(2):
        ax1.text(j, i, cm_base[i, j], ha="center", va="center", color="red", fontsize=14)

ax2 = fig3.add_subplot(132)
im2 = ax2.imshow(cm_pre, interpolation='nearest', cmap=plt.cm.Greens)
ax2.set_title('Pretrained Model CM')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
for i in range(2):
    for j in range(2):
        ax2.text(j, i, cm_pre[i, j], ha="center", va="center", color="red", fontsize=14)

ax3 = fig3.add_subplot(133)
fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
roc_auc_base = auc(fpr_base, tpr_base)
fpr_pre, tpr_pre, _ = roc_curve(y_test, y_prob_pre)
roc_auc_pre = auc(fpr_pre, tpr_pre)

ax3.plot(fpr_base, tpr_base, color='blue', lw=2, label=f'Base (AUC = {roc_auc_base:.3f})')
ax3.plot(fpr_pre, tpr_pre, color='green', lw=2, label=f'Pretrained (AUC = {roc_auc_pre:.3f})')
ax3.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
ax3.set_title('ROC Curves')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.legend(loc="lower right")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('3_model_comparison.png')
plt.show()

print("Результаты без предобучения:")
print("F1-мера:", f1_base)
print("Матрица ошибок:\n", cm_base)
print("\nРезультаты с предобучением:")
print("F1-мера:", f1_pre)
print("Матрица ошибок:\n", cm_pre)