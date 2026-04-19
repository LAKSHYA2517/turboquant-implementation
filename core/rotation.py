class RotationMatrix:
    @staticmethod
    def generate(d: int, seed: int = None) -> np.ndarray:
        """Generate orthogonal matrix via QR decomposition"""
        if seed: np.random.seed(seed)
        A = np.random.randn(d, d)
        Q, R = np.linalg.qr(A)
        return Q