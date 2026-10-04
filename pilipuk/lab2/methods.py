from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness

def run_pca(X, n_components=3):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    retained_var = pca.explained_variance_ratio_.sum()
    lost_var = 1.0 - retained_var
    return X_pca, retained_var, lost_var, pca.explained_variance_ratio_

def run_tsne(X, n_components=3, perplexity=30):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    kl_div = tsne.kl_divergence_
    trust = trustworthiness(X, X_tsne)
    return X_tsne, kl_div, trust