import torch
from loader import load_and_preprocess_data
from model import Autoencoder, train_autoencoder, evaluate_ae_reconstruction
from methods import run_pca, run_tsne
from vis import plot_projections_2d, plot_projections_3d

if __name__ == "__main__":
    file_path = "wdbc.data"
    
    X_scaled, y, dataloader, X_tensor = load_and_preprocess_data(file_path)
    input_dim = X_scaled.shape[1]
    
    X_pca_2d, pca_ret_2d, pca_lost_2d, pca_comp_2d = run_pca(X_scaled, n_components=2)
    X_pca_3d, pca_ret_3d, pca_lost_3d, pca_comp_3d = run_pca(X_scaled, n_components=3)
    
    print("\n--- PCA ---")
    print(f"2D Projection: Saved {pca_ret_2d*100:.2f}% | Lost {pca_lost_2d*100:.2f}%")
    print(f"3D Projection: Saved {pca_ret_3d*100:.2f}% | Lost {pca_lost_3d*100:.2f}%")

    ae_2d = train_autoencoder(Autoencoder(input_dim, 2), dataloader, epochs=100)
    ae_3d = train_autoencoder(Autoencoder(input_dim, 3), dataloader, epochs=100)
    
    X_ae_2d, ae_mse_2d, ae_ret_2d, ae_lost_2d = evaluate_ae_reconstruction(ae_2d, X_tensor)
    X_ae_3d, ae_mse_3d, ae_ret_3d, ae_lost_3d = evaluate_ae_reconstruction(ae_3d, X_tensor)
    
    print("\n--- Autoencoder ---")
    print(f"2D Latent: MSE recs = {ae_mse_2d:.4f} | Saved ~{ae_ret_2d*100:.2f}% | Lost ~{ae_lost_2d*100:.2f}%")
    print(f"3D Latent: MSE recs = {ae_mse_3d:.4f} | Saved ~{ae_ret_3d*100:.2f}% | Lost ~{ae_lost_3d*100:.2f}%")

    X_tsne_2d, tsne_kl_2d, tsne_trust_2d = run_tsne(X_scaled, n_components=2)
    X_tsne_3d, tsne_kl_3d, tsne_trust_3d = run_tsne(X_scaled, n_components=3)
    
    print("\n--- t-SNE ---")
    print(f"2D Projection: KL Divergence = {tsne_kl_2d:.4f} | Trustworthiness = {tsne_trust_2d*100:.2f}%")
    print(f"3D Projection: KL Divergence = {tsne_kl_3d:.4f} | Trustworthiness = {tsne_trust_3d*100:.2f}%")
    print("=" * 60)

    plot_projections_2d(X_pca_2d, X_tsne_2d, X_ae_2d, y)
    plot_projections_3d(X_pca_3d, X_tsne_3d, X_ae_3d, y)