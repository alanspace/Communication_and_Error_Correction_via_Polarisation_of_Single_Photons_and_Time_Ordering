# ==============================================================================
# functions.py
# A custom library for calculating communication protocol metrics.
# Now with GPU (PyTorch MPS) accelerated versions of key functions.
# ==============================================================================

import math
import numpy as np
from itertools import islice
from functools import lru_cache
import torch

# --- GPU / CPU Device Setup ---
# This block checks if you have a Mac with an MPS-enabled GPU and sets it as the default device.
# If not, it falls back to the CPU. Your main notebook can use this device variable.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("PyTorch MPS backend (GPU) is available and will be used.")
else:
    device = torch.device("cpu")
    print("PyTorch MPS backend not found. Using CPU.")

# ==============================================================================
# --- Generic Helper Function (for CPU) ---
# ==============================================================================

@lru_cache(maxsize=None)
def combination(n, r):
    """Computes 'n choose r' with caching for efficiency."""
    if r < 0 or r > n:
        return 0
    return math.factorial(int(n)) // (math.factorial(int(n - r)) * math.factorial(int(r)))

# ==============================================================================
# --- BPPM Protocol Functions ---
# ==============================================================================

# --- CPU Versions (for single values) ---

@lru_cache(maxsize=None)
def compute_N_BPPM(n_bppm):
    """Computes the number of time bins (N) for n_bppm photons (CPU)."""
    def agen():
        aset, sset, k = set(), set(), 0
        while True:
            k += 1
            while any(k + an in sset for an in aset): k += 1
            yield k; sset.update(k + an for an in aset); aset.add(k)
    a = list(islice(agen(), 100))
    compute_N = [sum(a[:i]) for i in range(1, len(a) + 1)]
    return compute_N[n_bppm - 1]

def p_err_BPPM(n, N, P_l, P_a, l, a):
    """Calculates single p(error) for BPPM (CPU)."""
    p_loss = combination(n, l) * (P_l**l) * ((1 - P_l)**(n - l))
    p_add = combination(N - n, a) * (P_a**a) * ((1 - P_a)**(N - n - a))
    return p_loss * p_add

def P_D_BPPM(n, P):
    """Calculates single P(Detection Failure) for BPPM (CPU)."""
    N = compute_N_BPPM(n)
    p_success_0 = p_err_BPPM(n, N, P, P, 0, 0)
    p_success_1_loss = p_err_BPPM(n, N, P, P, 1, 0)
    p_success_1_add = p_err_BPPM(n, N, P, P, 0, 1)
    return 1 - (p_success_0 + p_success_1_loss + p_success_1_add)

def I_AB_BPPM(n, P_D):
    """Calculates single Mutual Information for BPPM (CPU)."""
    if P_D >= 1: return 0
    num_codewords = math.factorial(n)
    H_initial = math.log2(num_codewords)
    return H_initial * (1 - P_D)

# --- GPU Versions (for arrays/tensors) ---

def P_D_BPPM_pt(n, P_tensor):
    """Calculates P(Detection Failure) for BPPM on a PyTorch tensor (GPU)."""
    N = compute_N_BPPM(n)
    # Move scalar values to the same device as the tensor
    n_t, N_t = torch.tensor(n, device=P_tensor.device), torch.tensor(N, device=P_tensor.device)
    
    # Use torch functions for vectorized operations on the GPU
    p_success_0 = combination(n,0) * (P_tensor**0) * ((1-P_tensor)**n) * \
                  combination(N-n,0) * (P_tensor**0) * ((1-P_tensor)**(N-n))
    p_success_1_loss = combination(n,1) * (P_tensor**1) * ((1-P_tensor)**(n-1)) * \
                       combination(N-n,0) * (P_tensor**0) * ((1-P_tensor)**(N-n))
    p_success_1_add = combination(n,0) * (P_tensor**0) * ((1-P_tensor)**n) * 
                      combination(N-n,1) * (P_tensor**1) * ((1-P_tensor)**(N-n-1))
    return 1 - (p_success_0 + p_success_1_loss + p_success_1_add)

def I_AB_BPPM_pt(n, P_D_tensor):
    """Calculates Mutual Information for BPPM on a PyTorch tensor (GPU)."""
    H_initial = math.log2(math.factorial(n))
    # Ensure P_D_tensor doesn't go above 1
    P_D_tensor = torch.clamp(P_D_tensor, max=1.0)
    return H_initial * (1 - P_D_tensor)

# ==============================================================================
# --- PPM Protocol Functions ---
# ==============================================================================

# --- CPU Versions ---
def P_D_PPM(n, M, P):
    p_success = combination(n, 0) * (P**0) * ((1-P)**n) * \
                combination(M-n, 0) * (P**0) * ((1-P)**(M-n))
    return 1 - p_success

def I_AB_PPM(n, M, P_D):
    if P_D >= 1 or M <= 1: return 0
    return math.log2(M) * (1 - P_D)

# --- GPU Versions ---
def P_D_PPM_pt(n, M_tensor, P_tensor):
    """Calculates P(D) for PPM on a PyTorch tensor (GPU)."""
    p_success = combination(n, 0) * (P_tensor**0) * ((1-P_tensor)**n) * \
                combination(M_tensor-n, 0) * (P_tensor**0) * ((1-P_tensor)**(M_tensor-n))
    return 1 - p_success

def I_AB_PPM_pt(n, M_tensor, P_D_tensor):
    """Calculates I(A;B) for PPM on a PyTorch tensor (GPU)."""
    # Use torch.log2 for element-wise log on the tensor
    H_initial = torch.log2(M_tensor)
    P_D_tensor = torch.clamp(P_D_tensor, max=1.0)
    return H_initial * (1 - P_D_tensor)
    
# ==============================================================================
# --- OOK Protocol Functions ---
# ==============================================================================

# ==============================================================================
# --- OOK (On-Off Keying) Protocol Functions ---
# ==============================================================================

# --- CPU Version (for single values) ---

def I_AB_OOK(n_ook, p_flip):
    """
    Calculates single I(A;B) for OOK (CPU).
    This analytical solution is much faster than building the transition matrix.
    """
    if p_flip == 0:
        return n_ook  # Perfect channel, I(A;B) = H(Y) = n_ook
    if p_flip >= 0.5:
        return 0  # Completely noisy channel, no information can be transmitted
        
    # H_bit_flip is the binary entropy function H(p)
    H_bit_flip = -p_flip * math.log2(p_flip) - (1 - p_flip) * math.log2(1 - p_flip)
    
    # For n_ook independent bits, the conditional entropy H(Y|X) = n_ook * H(p)
    H_Y_given_X = n_ook * H_bit_flip
    
    # H(Y) assuming uniform input is n_ook
    H_Y = n_ook
    
    # I(X;Y) = H(Y) - H(Y|X)
    return H_Y - H_Y_given_X

# --- GPU Version (for arrays/tensors) ---

def I_AB_OOK_pt(n_ook, p_flip_tensor):
    """
    Calculates I(A;B) for OOK on a PyTorch tensor (GPU).
    This is the vectorized version of the CPU function.
    
    Args:
        n_ook (int): The number of bits in the OOK symbol.
        p_flip_tensor (torch.Tensor): A tensor of bit-flip probabilities.
    """
    # Create tensors for edge cases and results on the same device
    zero_tensor = torch.tensor(0.0, device=p_flip_tensor.device)
    n_ook_tensor = torch.tensor(float(n_ook), device=p_flip_tensor.device)

    # Handle edge cases to avoid log(0)
    p_safe = torch.clamp(p_flip_tensor, min=1e-9, max=1.0 - 1e-9)
    
    # Calculate binary entropy H(p) for the entire tensor
    H_bit_flip = -p_safe * torch.log2(p_safe) - (1 - p_safe) * torch.log2(1 - p_safe)
    
    # H(Y|X) = n * H(p)
    H_Y_given_X = n_ook_tensor * H_bit_flip
    
    # H(Y) = n
    H_Y = n_ook_tensor
    
    # I(X;Y) = H(Y) - H(Y|X)
    mutual_info = H_Y - H_Y_given_X
    
    # Ensure that for p >= 0.5, the mutual information is 0
    # Create a mask where p_flip_tensor is >= 0.5
    mask = p_flip_tensor >= 0.5
    # Apply the mask: where the condition is true, set MI to 0
    mutual_info[mask] = 0.0

    return mutual_info