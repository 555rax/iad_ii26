import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

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
base_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
y_pred_base = (base_model.predict(X_test) > 0.5).astype(int)
f1_base = f1_score(y_test, y_pred_base)
cm_base = confusion_matrix(y_test, y_pred_base)

def build_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(inputs)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    outputs = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(inputs=inputs, outputs=outputs)
    encoder = Model(inputs=inputs, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder(X_train.shape[1])
autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

for layer in encoder.layers:
    layer.trainable = False

pretrained_input = Input(shape=(X_train.shape[1],))
pretrained_x = encoder(pretrained_input)
pretrained_x = Dense(16, activation='relu')(pretrained_x)
pretrained_output = Dense(1, activation='sigmoid')(pretrained_x)
pretrained_model = Model(inputs=pretrained_input, outputs=pretrained_output)
pretrained_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
pretrained_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

y_pred_pretrained = (pretrained_model.predict(X_test) > 0.5).astype(int)
f1_pretrained = f1_score(y_test, y_pred_pretrained)
cm_pretrained = confusion_matrix(y_test, y_pred_pretrained)

print("Результаты без предобучения:")
print("F1-мера:", f1_base)
print("Матрица ошибок:\n", cm_base)
print("\nРезультаты с предобучением:")
print("F1-мера:", f1_pretrained)
print("Матрица ошибок:\n", cm_pretrained)