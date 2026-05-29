import torch
from .darkdiv import compute_dark_diversity

# Print CUDA availability on import
if torch.cuda.is_available():
    print(f"pmf-dark: CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
else:
    print("pmf-dark: CUDA is not available. Using CPU.")

__all__ = ["compute_dark_diversity"]
