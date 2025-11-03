import numpy as np
from sklearn.decomposition import PCA


def iterative_pca_imputation(
    train_surfaces: np.ndarray,
    train_masks: np.ndarray,
    n_components: int,
    max_iter: int = 10,
    tol: float = 1e-4,
) -> PCA:
    """
    Fits a PCA model on incomplete data using iterative imputation.
    """
    n_samples, h, w = train_surfaces.shape
    flat_surfaces = train_surfaces.reshape(n_samples, -1)
    flat_masks = train_masks.reshape(n_samples, -1)

    # Initial imputation with the mean of observed values
    mean_val = np.mean(flat_surfaces[flat_masks])
    imputed_surfaces = np.where(flat_masks, flat_surfaces, mean_val)

    pca = PCA(n_components=n_components)

    for i in range(max_iter):
        prev_imputed = np.copy(imputed_surfaces)

        # Fit PCA on the current imputed data
        transformed = pca.fit_transform(imputed_surfaces)

        # Reconstruct
        reconstructed = pca.inverse_transform(transformed)

        # Update only the missing values
        imputed_surfaces[~flat_masks] = reconstructed[~flat_masks]

        # Check for convergence
        change = np.sum((imputed_surfaces - prev_imputed) ** 2) / np.sum(
            prev_imputed**2
        )
        if change < tol:
            break

    return pca


def apply_pca_reconstruction(
    surfaces: np.ndarray, masks: np.ndarray, pca: PCA
) -> np.ndarray:
    """
    Reconstructs surfaces using a trained PCA model.
    """
    n_samples, h, w = surfaces.shape
    flat_surfaces = surfaces.reshape(n_samples, -1)
    flat_masks = masks.reshape(n_samples, -1)

    # Impute with mean from training data (stored in PCA)
    imputed_surfaces = np.where(flat_masks, flat_surfaces, pca.mean_)

    transformed = pca.transform(imputed_surfaces)
    reconstructed = pca.inverse_transform(transformed)

    return reconstructed.reshape(n_samples, h, w)
