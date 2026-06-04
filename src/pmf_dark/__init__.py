import torch
from .darkdiv import compute_dark_diversity, PMFDark

# Print CUDA availability on import
if torch.cuda.is_available():
    print(f"pmf-dark: CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
else:
    print("pmf-dark: CUDA is not available. Using CPU.")

# Lazy loading of datasets
_cached_datasets = {}


def __getattr__(name):
    if name in ("env", "survey"):
        if name not in _cached_datasets:
            import importlib.resources
            import pandas as pd

            ref = importlib.resources.files("pmf_dark").joinpath(f"data/{name}.csv")
            with importlib.resources.as_file(ref) as path:
                _cached_datasets[name] = pd.read_csv(path)
        return _cached_datasets[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + ["env", "survey"])


__all__ = ["compute_dark_diversity", "PMFDark", "env", "survey"]
