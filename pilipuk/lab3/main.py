import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

def run_pipeline(name, X, y):
    print(f"\n==================== DATASET: {name} ====================")
    
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    if len(cat_cols) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
            ]
        )
        X_processed = preprocessor.fit_transform(X)
    else:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    input_dim = X_train.shape[1]
    
    def create_classifier(dim):
        model = models.Sequential([
            layers.Input(shape=(dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
        
    model_scratch = create_classifier(input_dim)
    model_scratch.compile(
        optimizer=optimizers.Adam(learning_rate=0.005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model_scratch.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    preds_scratch = (model_scratch.predict(X_test, verbose=0) >= 0.5).astype(int)
    
    print("=== WITHOUT PRETRAINING ===")
    print("Accuracy:", accuracy_score(y_test, preds_scratch))
    print("F1-Score:", f1_score(y_test, preds_scratch))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds_scratch))
    print("Classification Report:\n", classification_report(y_test, preds_scratch))
    
    def train_autoencoder(in_dim, out_dim, data, epochs=30):
        ae_input = layers.Input(shape=(in_dim,))
        encoded = layers.Dense(out_dim, activation='relu')(ae_input)
        decoded = layers.Dense(in_dim, activation='linear')(encoded)
        ae = models.Model(ae_input, decoded)
        ae.compile(optimizer=optimizers.Adam(learning_rate=0.005), loss='mse')
        ae.fit(data, data, epochs=epochs, batch_size=32, verbose=0)
        return models.Model(ae_input, encoded)
        
    enc1 = train_autoencoder(input_dim, 64, X_train, epochs=30)
    data1 = enc1.predict(X_train, verbose=0)
    
    enc2 = train_autoencoder(64, 32, data1, epochs=30)
    data2 = enc2.predict(data1, verbose=0)
    
    enc3 = train_autoencoder(32, 16, data2, epochs=30)
    
    model_pretrained = create_classifier(input_dim)
    model_pretrained.layers[0].set_weights(enc1.layers[1].get_weights())
    model_pretrained.layers[1].set_weights(enc2.layers[1].get_weights())
    model_pretrained.layers[2].set_weights(enc3.layers[1].get_weights())
    
    model_pretrained.compile(
        optimizer=optimizers.Adam(learning_rate=0.005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model_pretrained.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    preds_pt = (model_pretrained.predict(X_test, verbose=0) >= 0.5).astype(int)
    
    print("\n=== WITH AUTOENCODER PRETRAINING ===")
    print("Accuracy:", accuracy_score(y_test, preds_pt))
    print("F1-Score:", f1_score(y_test, preds_pt))
    cm = confusion_matrix(y_test, preds_pt)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, preds_pt))
    
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({name})')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [0, 1])
    plt.yticks(tick_marks, [0, 1])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()

url_credit = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
col_names_credit = [f"A{i}" for i in range(1, 16)] + ["target"]
df_credit = pd.read_csv(url_credit, names=col_names_credit, na_values="?").dropna()
X_credit = df_credit.drop(columns=["target"])
y_credit = df_credit["target"].map({"+": 1, "-": 0}).values
run_pipeline("Credit Approval", X_credit, y_credit)

url_wdbc = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
df_wdbc = pd.read_csv(url_wdbc, header=None)
y_wdbc = df_wdbc.iloc[:, 1].map({'M': 1, 'B': 0}).values
X_wdbc = df_wdbc.drop(columns=[0, 1])
run_pipeline("Breast Cancer Wisconsin", X_wdbc, y_wdbc)