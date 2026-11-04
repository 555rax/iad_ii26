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

def train_rbm(visible_dim, hidden_dim, data, epochs=30, lr=0.01, batch_size=32):
    W = np.random.normal(0.0, 0.1, (visible_dim, hidden_dim))
    a = np.zeros(visible_dim)
    b = np.zeros(hidden_dim)
    n_samples = data.shape[0]
    for epoch in range(epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            v0 = data[batch_indices]
            h0_prob = 1 / (1 + np.exp(-(np.dot(v0, W) + b)))
            h0_states = (h0_prob > np.random.rand(*h0_prob.shape)).astype(float)
            v1_prob = 1 / (1 + np.exp(-(np.dot(h0_states, W.T) + a)))
            v1_states = (v1_prob > np.random.rand(*v1_prob.shape)).astype(float)
            h1_prob = 1 / (1 + np.exp(-(np.dot(v1_states, W) + b)))
            pos_associations = np.dot(v0.T, h0_prob)
            neg_associations = np.dot(v1_states.T, h1_prob)
            W += lr * (pos_associations - neg_associations) / v0.shape[0]
            a += lr * np.mean(v0 - v1_states, axis=0)
            b += lr * np.mean(h0_prob - h1_prob, axis=0)
    return W, b

def rbm_transform(data, W, b):
    return 1 / (1 + np.exp(-(np.dot(data, W) + b)))

def run_pipeline(name, X, y):
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
    cm_scratch = confusion_matrix(y_test, preds_scratch)
    
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
    
    model_pretrained_ae = create_classifier(input_dim)
    model_pretrained_ae.layers[0].set_weights(enc1.layers[1].get_weights())
    model_pretrained_ae.layers[1].set_weights(enc2.layers[1].get_weights())
    model_pretrained_ae.layers[2].set_weights(enc3.layers[1].get_weights())
    model_pretrained_ae.compile(
        optimizer=optimizers.Adam(learning_rate=0.005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model_pretrained_ae.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    preds_pt_ae = (model_pretrained_ae.predict(X_test, verbose=0) >= 0.5).astype(int)
    cm_ae = confusion_matrix(y_test, preds_pt_ae)

    W1, b1 = train_rbm(input_dim, 64, X_train, epochs=30)
    rbm_data1 = rbm_transform(X_train, W1, b1)
    W2, b2 = train_rbm(64, 32, rbm_data1, epochs=30)
    rbm_data2 = rbm_transform(rbm_data1, W2, b2)
    W3, b3 = train_rbm(32, 16, rbm_data2, epochs=30)
    
    model_pretrained_rbm = create_classifier(input_dim)
    model_pretrained_rbm.layers[0].set_weights([W1, b1])
    model_pretrained_rbm.layers[1].set_weights([W2, b2])
    model_pretrained_rbm.layers[2].set_weights([W3, b3])
    model_pretrained_rbm.compile(
        optimizer=optimizers.Adam(learning_rate=0.005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model_pretrained_rbm.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    preds_pt_rbm = (model_pretrained_rbm.predict(X_test, verbose=0) >= 0.5).astype(int)
    cm_rbm = confusion_matrix(y_test, preds_pt_rbm)
    
    results_df = pd.DataFrame({
        "Model": ["Scratch", "Autoencoder", "RBM"],
        "Accuracy": [
            accuracy_score(y_test, preds_scratch),
            accuracy_score(y_test, preds_pt_ae),
            accuracy_score(y_test, preds_pt_rbm)
        ],
        "F1-Score": [
            f1_score(y_test, preds_scratch),
            f1_score(y_test, preds_pt_ae),
            f1_score(y_test, preds_pt_rbm)
        ]
    })
    
    print(f"\n==================== DATASET: {name} ====================")
    print(results_df.to_string(index=False))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cms = [cm_scratch, cm_ae, cm_rbm]
    titles = ["Scratch", "Autoencoder", "RBM"]
    
    for ax, cm, title in zip(axes, cms, titles):
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{title} ({name})')
        fig.colorbar(im, ax=ax)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([0, 1])
        ax.set_yticklabels([0, 1])
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > cm.max() / 2. else "black")
                        
    plt.tight_layout()
    plt.show()

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