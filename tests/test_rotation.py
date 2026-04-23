import numpy as np
from core.rotation import RotationMatrix

def test_rotation_is_orthogonal():
    """Verify rotation matrix is orthogonal: R^T @ R = I"""
    d = 20
    R = RotationMatrix.generate(d, seed=42)
    
    # Check orthogonality
    I = R.T @ R
    assert np.allclose(I, np.eye(d)), "Not orthogonal!"

def test_rotation_determinant():
    """Verify determinant is 1 (proper rotation, not reflection)"""
    R = RotationMatrix.generate(50, seed=42)
    det = np.linalg.det(R)
    assert np.isclose(det, 1.0, atol=1e-6), f"Det = {det}, expected 1"