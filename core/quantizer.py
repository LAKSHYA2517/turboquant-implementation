import numpy as np
try:
    from .rotation import RotationMatrix
except ImportError:
    from rotation import RotationMatrix

class TurboQuantMSE:
    """MSE-optimal quantization as per Paper Algorithm 1"""
    
    def __init__(self, d: int, bitwidth: int, seed: int = 42):
        self.d = d
        self.b = bitwidth
        self.rotation = RotationMatrix.generate(d, seed=seed)
        self.centroids = self._get_codebook(bitwidth)
    
    def _get_codebook(self, bitwidth: int) -> np.ndarray:
        
        if bitwidth == 1:
            # Optimal 1-bit centroids: ±√(2/(πd))
            c = np.sqrt(2 / (np.pi * self.d))
            return np.array([-c, c])
        
        elif bitwidth == 2:
            # Approximate optimal for b=2
            return np.array([
                -0.951 / np.sqrt(self.d),
                -0.453 / np.sqrt(self.d),
                0.453 / np.sqrt(self.d),
                0.951 / np.sqrt(self.d)
            ])
        
        else:
            # For b>=3: uniform initialization
            # (In production: precompute with Lloyd-Max)
            n_levels = 2 ** bitwidth
            return np.linspace(-0.99, 0.99, n_levels)
    
    def quantize(self, x: np.ndarray) -> tuple:

        # Normalize
        norm = np.linalg.norm(x)
        x_normalized = x / (norm + 1e-10)
        
        # Rotate
        y = self.rotation @ x_normalized
        
        # Quantize: find nearest centroid for each coordinate
        indices = np.array([
            np.argmin(np.abs(y[i] - self.centroids))
            for i in range(self.d)
        ], dtype=np.int32)
        
        return indices, norm
    
    def dequantize(self, indices: np.ndarray, norm: float) -> np.ndarray:

        # Look up centroids
        y_reconstructed = self.centroids[indices.astype(int)]
        
        # Rotate back
        x_normalized_reconstructed = self.rotation.T @ y_reconstructed
        
        # Normalize to unit norm, then rescale by original norm
        norm_reconstructed = np.linalg.norm(x_normalized_reconstructed)
        x_normalized_reconstructed = x_normalized_reconstructed / (norm_reconstructed + 1e-10)
        x_reconstructed = x_normalized_reconstructed * norm
        
        return x_reconstructed