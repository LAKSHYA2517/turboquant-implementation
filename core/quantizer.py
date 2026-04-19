class TurboQuantMSE:
    def __init__(self, d: int, bitwidth: int):
        self.rotation = RotationMatrix.generate(d)
        self.centroids = self._get_codebook(bitwidth)
    
    def quantize(self, x: np.ndarray) -> np.ndarray:
        y = self.rotation @ x
        return np.array([np.argmin(np.abs(y - c)) for c in self.centroids])
    
    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        y_recon = self.centroids[indices]
        return self.rotation.T @ y_recon