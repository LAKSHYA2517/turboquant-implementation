import numpy as np

class RotationMatrix:
    """Random orthogonal matrix via QR decomposition"""
    
    @staticmethod
    def generate(d: int, seed: int = None) -> np.ndarray:
        """
        Generate random rotation matrix
        
        Paper Section 3.1: We use QR decomposition of Gaussian matrix
        This gives uniform distribution over rotation group SO(d)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Random Gaussian matrix
        A = np.random.randn(d, d)
        
        # QR decomposition
        Q, R = np.linalg.qr(A)
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        
        return Q