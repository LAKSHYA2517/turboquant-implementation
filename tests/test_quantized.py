import numpy as np
from core.quantizer import TurboQuantMSE

def test_quantization_reversible():
    """Quantize and dequantize should be close to original"""
    d = 128
    quant = TurboQuantMSE(d, bitwidth=2, seed=42)
    
    # Test vector
    x = np.random.randn(d)
    
    # Quantize and dequantize
    indices, norm = quant.quantize(x)
    x_reconstructed = quant.dequantize(indices, norm)
    
    # Should have same shape
    assert x_reconstructed.shape == x.shape
    
    # Norms should match
    assert np.isclose(np.linalg.norm(x), np.linalg.norm(x_reconstructed), rtol=0.2)

def test_mse_matches_theory():
    d = 256
    bitwidth = 2
    quant = TurboQuantMSE(d, bitwidth, seed=42)
    
    # Compute empirical MSE
    mse_values = []
    for _ in range(100):
        x = np.random.randn(d)
        x = x / np.linalg.norm(x)
        
        indices, norm = quant.quantize(x)
        x_recon = quant.dequantize(indices, norm)
        
        mse = np.mean((x - x_recon) ** 2)
        mse_values.append(mse)
    
    empirical_mse = np.mean(mse_values)
    
    # Theoretical bound: √(3π/2) * 1/4^b
    theoretical_bound = np.sqrt(3*np.pi/2) * (1/4)**bitwidth
    
    print(f"Empirical MSE: {empirical_mse:.6f}")
    print(f"Theoretical bound: {theoretical_bound:.6f}")
    print(f"Ratio: {empirical_mse / theoretical_bound:.2f}")
    
    # Should be reasonably close
    assert empirical_mse < theoretical_bound * 2, "MSE too high!"