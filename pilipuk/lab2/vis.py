import matplotlib.pyplot as plt
import seaborn as sns

def plot_projections_2d(X_pca, X_tsne, X_ae, labels):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, ax=axes[0], palette="tab10", alpha=0.8)
    axes[0].set_title("PCA 2D Projection")
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, ax=axes[1], palette="tab10", alpha=0.8)
    axes[1].set_title("t-SNE 2D Projection")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    
    sns.scatterplot(x=X_ae[:, 0], y=X_ae[:, 1], hue=labels, ax=axes[2], palette="tab10", alpha=0.8)
    axes[2].set_title("Autoencoder 2D Latent Space")
    axes[2].set_xlabel("Latent 1")
    axes[2].set_ylabel("Latent 2")
    
    plt.tight_layout()
    plt.show()

def plot_projections_3d(X_pca, X_tsne, X_ae, labels):
    fig = plt.figure(figsize=(18, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='tab10', alpha=0.8)
    ax1.set_title("PCA 3D Projection")
    ax1.set_xlabel("PC 1")
    ax1.set_ylabel("PC 2")
    ax1.set_zlabel("PC 3")
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=labels, cmap='tab10', alpha=0.8)
    ax2.set_title("t-SNE 3D Projection")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.set_zlabel("t-SNE 3")
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(X_ae[:, 0], X_ae[:, 1], X_ae[:, 2], c=labels, cmap='tab10', alpha=0.8)
    ax3.set_title("Autoencoder 3D Latent Space")
    ax3.set_xlabel("Latent 1")
    ax3.set_ylabel("Latent 2")
    ax3.set_zlabel("Latent 3")
    
    plt.tight_layout()
    plt.show()