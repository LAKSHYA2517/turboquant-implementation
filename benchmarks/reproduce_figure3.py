import numpy as np
import matplotlib.pyplot as plt
from core.quantizer import TurboQuantMSE

def reproduce_figure_3(d: int = 256, n_trials: int = 100):
    bitwidths = [1, 2, 3, 4]
    mse_empirical = []
    mse_theory = []
    
    for b in bitwidths:
        print(f"Testing bitwidth {b}...")
        
        quant = TurboQuantMSE(d, b, seed=42)
        mse_values = []
        
        for trial in range(n_trials):
            # Random unit vector
            x = np.random.randn(d)
            x = x / np.linalg.norm(x)
            
            # Quantize and reconstruct
            indices, norm = quant.quantize(x)
            x_recon = quant.dequantize(indices, norm)
            
            # MSE
            mse = np.mean((x - x_recon) ** 2)
            mse_values.append(mse)
        
        avg_mse = np.mean(mse_values)
        mse_empirical.append(avg_mse)
        
        # Theoretical bound
        theory = np.sqrt(3*np.pi/2) * (1/4)**b
        mse_theory.append(theory)
        
        print(f"  Empirical MSE: {avg_mse:.6f}")
        print(f"  Theoretical: {theory:.6f}")
        print(f"  Ratio: {avg_mse / theory:.2f}x")
    
    # Plot (matches Paper Figure 3)
    plt.figure(figsize=(8, 6))
    plt.semilogy(bitwidths, mse_empirical, 'o-', label='Empirical (Your Implementation)', 
                 linewidth=2, markersize=8)
    plt.semilogy(bitwidths, mse_theory, 's--', label='Theoretical Bound (Paper Theorem 1)', 
                 linewidth=2, markersize=8)
    
    plt.xlabel('Bit-width (b)', fontsize=12)
    plt.ylabel('MSE Distortion', fontsize=12)
    plt.title(f'TurboQuant: MSE vs Bit-width (d={d})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmarks/results/mse_vs_bitwidth.png', dpi=300)
    print(f"\nPlot saved to benchmarks/results/mse_vs_bitwidth.png")
    
    return mse_empirical, mse_theory

if __name__ == '__main__':
    reproduce_figure_3()