import numpy as np

class GaussianNoiseAugmenter:
    """
    Data augmentation using white Gaussian noise
    Multiplies dataset by adding noisy versions
    """
    
    def __init__(self, noise_levels=[0.10], random_state=42):
        """
        Args:
            noise_levels: List of noise percentages (e.g., [0.05, 0.10, 0.20])
            random_state: Random seed for reproducibility
        """
        self.noise_levels = noise_levels
        self.random_state = random_state
        np.random.seed(random_state)
    
    def add_gaussian_noise(self, data, noise_level):
        """
        Add white Gaussian noise to data
        
        Args:
            data: numpy array of shape (n_samples, n_features)
            noise_level: percentage of noise (e.g., 0.05 for 5%)
        
        Returns:
            Noisy data array
        """
        noisy_data = data.copy()
        
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            col_std = np.std(col_data)
            noise = np.random.normal(0, noise_level * col_std, size=col_data.shape)
            noisy_data[:, col_idx] = col_data + noise
        
        return noisy_data
    
    def augment_data(self, X, y, verbose=True):
        """
        Augment data by adding multiple noisy versions
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            verbose: Print augmentation info
        
        Returns:
            X_augmented, y_augmented: Augmented arrays (4× original size)
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"DATA AUGMENTATION WITH WHITE GAUSSIAN NOISE")
            print(f"{'='*80}")
            print(f"Original dataset size: {X.shape[0]} samples")
            print(f"Noise levels: {[f'{level*100:.0f}%' for level in self.noise_levels]}")
        
        # Start with original data
        X_list = [X]
        y_list = [y]
        
        # Add noisy versions
        for noise_level in self.noise_levels:
            X_noisy = self.add_gaussian_noise(X, noise_level)
            
            if verbose:
                noise_magnitude = np.mean(np.abs(X_noisy - X))
                print(f"  - Added {noise_level*100:.0f}% noise version (mean noise: {noise_magnitude:.4f})")
            
            X_list.append(X_noisy)
            y_list.append(y.copy())
        
        # Concatenate all versions
        X_augmented = np.vstack(X_list)
        y_augmented = np.hstack(y_list)
        
        if verbose:
            print(f"Final augmented dataset size: {X_augmented.shape[0]} samples ({X_augmented.shape[0] / X.shape[0]:.1f}x)")
            print(f"{'='*80}\n")
        
        return X_augmented, y_augmented


def augment_training_data(X_train, y_train, noise_levels=[0.05, 0.10, 0.20], verbose=True):
    """
    Convenience function to augment training data
    
    Args:
        X_train: Training features
        y_train: Training targets
        noise_levels: List of noise percentages
        verbose: Print info
    
    Returns:
        X_train_augmented, y_train_augmented
    """
    augmenter = GaussianNoiseAugmenter(noise_levels=noise_levels)
    return augmenter.augment_data(X_train, y_train, verbose=verbose)
