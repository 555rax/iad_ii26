import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

def train_autoencoder(model, dataloader, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
    return model

def evaluate_ae_reconstruction(model, X_tensor):
    model.eval()
    with torch.no_grad():
        reconstructed, latent = model(X_tensor)
        mse_loss = nn.MSELoss()(reconstructed, X_tensor).item()
        
    retained_info = max(0.0, 1.0 - mse_loss)
    lost_info = mse_loss
    return latent.numpy(), mse_loss, retained_info, lost_info