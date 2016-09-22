"""
Calculate Strehl ratio. 
Load amplitude files 
"""
import numpy as np
import cmath 

def Strehl(A_complex):
    """
    Strehl ratio 
    """
    amp_real = np.abs(A_complex)
#     phase_real = cmath.phase(A_complex)
    
    strehl = np.abs(A_complex.sum())**2/(amp_real**2*A_complex.size)
    return strehl
    
