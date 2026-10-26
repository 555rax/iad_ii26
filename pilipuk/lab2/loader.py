import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path, batch_size=32):
    df = pd.read_csv(file_path, header=None)
    X = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return X_scaled, y_encoded, dataloader, X_tensor