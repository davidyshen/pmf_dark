# PMF-dark: Using matrix factorisation for dark diversity estimation

[GitHub](https://github.com/davidyshen/pmf_dark/) repository

## Overview

This repository implements **PMF-dark** using Bayesian Probabilistic Matrix Factorisation to estimate **dark diversity** - the set of species absent from a site despite having suitable environmental conditions. The method uses **counterfactual predictions** to reconstruct the potential species pool by separating environmental effects from unmeasured drivers of absence (e.g., land-use degradation, dispersal limitation, biotic interactions).

## Installation

### Requirements

- Python 3.12, 3.13, or 3.14
- PyTorch (with CUDA support if using GPU)
- Pyro (pyro-ppl)
- Pandas
- NumPy
- Matplotlib

### 1. Standard Installation (from PyPI)

If you just want to use the package in your own projects, you can install it directly:

```bash
pip install pmf-dark
```

> [!NOTE]
> Installing `pmf-dark` automatically installs a default PyPI version of PyTorch (typically a CPU-only build on Windows). If you need CUDA GPU support, follow the CUDA replacement steps below.

### 2. CUDA GPU Support (Optional)

To run computations on an NVIDIA GPU, you must manually install or replace PyTorch with the correct CUDA-enabled build.

Visit the [PyTorch Getting Started guide](https://pytorch.org/get-started/locally/) to select the correct command for your CUDA version and OS. For example, to install or switch to PyTorch with CUDA 12.4 support:

```bash
# Uninstall standard/CPU torch first to prevent conflicts
pip uninstall -y torch torchvision torchaudio

# Install CUDA-enabled torch
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Or force reinstall to switch from CPU to CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force
```

---

### 3. Development & Demo Setup (from Source)

If you cloned this repository to run the demo notebooks (`demo.ipynb`) or want to make changes to the source code:

#### Step A: Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/davidyshen/PMF_dark.git
cd PMF_dark

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Step B: Install PyTorch (CUDA or CPU)

If you want GPU acceleration, install the CUDA version first (following the **CUDA GPU Support** instructions above). Otherwise, you can skip this step.

#### Step C: Install the Package in Editable Mode

This installs the local package and automatically resolves all remaining dependencies:

If installing via pip:

```bash
pip install -e .
```

If using Poetry:

```bash
poetry install
```

## Usage

`pmf_dark` provides a flexible Python API to fit models, generate predictions, and estimate dark diversity.

### Quick Start: Basic API Usage (Recommended)

`pmf_dark` provides a clean object-oriented interface (`PMFDark` model class) to fit models and easily retrieve different predictions (current distribution, potential species pool, and dark diversity) without redundant model refitting.

```python
from pmf_dark import PMFDark, env, survey

# 1. Initialize the PMFDark model
model = PMFDark(
    model_type="gaussian",   # Ecological response model: "linear" | "gaussian" | "bnn"
    method="svi",            # Inference method: "svi" | "mcmc"
    num_factors=2,           # Number of latent factors
)

# 2. Fit the model once using the built-in demo datasets (env and survey)
## env contains the column landuse, which is a driver of dark diversity. We need to drop it from the modelling data
env = env.drop(columns=["landuse"])
model.fit(
    y=survey,                # Loaded directly as a pandas DataFrame
    x=env,                   # Loaded directly as a pandas DataFrame
    num_iterations=2500,     # SVI parameter
    categorical_cols=["landuse"] # Explicitly treat landuse as categorical
)

# 3. Generate predictions from the fitted model
p_dist = model.distribution()  # Current species distribution (with latent factors)
p_pool = model.pool()          # Potential species pool (counterfactual / env only)
p_dark = model.dark()          # Dark diversity (pool prediction where species is not observed, NaN otherwise)
```

For backward compatibility, a functional API `compute_dark_diversity` is also provided.

### `PMFDark` Class API (Recommended)

#### 1. Constructor: `PMFDark()`

Configures the core model architecture and inference settings:

```python
model = PMFDark(
    model_type="gaussian",  # Response model: "linear" | "gaussian" | "bnn"
    num_factors=1,          # Number of latent factors for residual covariance
    method="svi",           # Inference method: "svi" | "mcmc"
    cuda=False,             # GPU computation (SVI only)
    **kwargs                # Extra response model specific arguments
)
```

#### 2. Fitting: `model.fit()`

Trains model parameters on the provided dataset:

```python
model.fit(
    y,                      # Species presence-absence/count matrix (n_sites, n_species)
    x,                      # Environmental predictor matrix (n_sites, n_env)
    categorical_cols=None,  # Explicit list of columns to treat as categorical
    batch_size=None,        # Mini-batch size (SVI only)
    **kwargs                # Hyperparameters for SVI (num_iterations, lr) or MCMC (num_chains)
)
```

#### 3. Prediction: `model.distribution()`, `model.pool()`, and `model.dark()`

Generates species occurrence probabilities:

```python
p_dist = model.distribution(pred_batch_size=None, return_means=True)
p_pool = model.pool(pred_batch_size=None, return_means=True)
p_dark = model.dark(pred_batch_size=None, return_means=True)
```

*   **`pred_batch_size`** *(int, optional)*: Chunk size to process sites during prediction (useful to manage memory consumption).
*   **`return_means`** *(bool, default `True`)*: Returns a `pandas.DataFrame` of posterior means if `True`, or a NumPy `ndarray` of raw posterior samples with shape `(num_samples, n_sites, n_species)` if `False`.

---

### `compute_dark_diversity()` Function Arguments

```python
compute_dark_diversity(
    y,                      # Species presence-absence/count matrix (n_sites, n_species)
    x,                      # Environmental predictor matrix (n_sites, n_env)
    model_type="gaussian",  # "linear" | "gaussian" | "bnn"
    num_factors=1,          # Number of latent factors for residual covariance
    method="svi",           # "svi" | "mcmc"
    cuda=False,             # GPU computation (SVI only)
    include_latent=True,    # Include latent factors in predictions
    return_means=True,      # Return means or full posterior samples
    batch_size=None,        # Mini-batch size for SVI training (default: None)
    pred_batch_size=None,   # Site-chunk size for prediction output (default: None)
    categorical_cols=None,  # Explicit list of columns to treat as categorical variables
    **kwargs,               # Extra model/method specific arguments
)
```

#### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | array-like | — | Species matrix (presence/absence or counts) with shape `(n_sites, n_species)`. |
| `x` | array-like | — | Environmental predictor matrix with shape `(n_sites, n_env)`. |
| `model_type` | str | `"gaussian"` | Ecological response model: `"linear"`, `"gaussian"` (quadratic niche), or `"bnn"` (Bayesian Neural Network). |
| `num_factors` | int | `1` | Number of latent factors used to model residual species covariance. |
| `method` | str | `"svi"` | Inference method: `"svi"` (Stochastic Variational Inference) or `"mcmc"` (Hamiltonian NUTS). |
| `cuda` | bool | `False` | Use GPU computation (SVI only, requires CUDA-enabled PyTorch build). |
| `include_latent` | bool | `True` | Include latent factors when computing predictions (Full predictions). Set `False` for counterfactual (environment-only) predictions. |
| `return_means` | bool | `True` | Return posterior means (`True`) or full posterior samples (`False`). |
| `batch_size` | int | `None` | Mini-batch size for SVI training (None fits all data in one step). |
| `pred_batch_size` | int | `None` | Site-chunk size for prediction output (None uses full-batch prediction). |
| `categorical_cols` | list | `None` | Explicit list of column names in `x` to treat as categorical variables (e.g. label-encoded integers). |

#### Method-Specific Arguments (`**kwargs`)

- **SVI (`method="svi"`)**:
  - `num_iterations=2500`: Number of training steps.
  - `lr=0.01`: Adam optimizer learning rate.
  - `num_samples=1000`: Number of posterior samples to draw for predictions.
- **MCMC (`method="mcmc"`)**:
  - `num_samples=1000`: Number of posterior samples.
  - `warmup_steps=500`: Warmup (burn-in) steps for NUTS.

---

### Ecological Response Models

#### 1. Linear Model (`model_type="linear"`)

Models species responses linearly (on the logit scale). Simplest model.

```python
p_linear = compute_dark_diversity(
    y, x,
    model_type="linear",
    method="svi",
    num_iterations=2000
)
```

#### 2. Gaussian Niche Model (`model_type="gaussian"`)

Models symmetric, bell-shaped (quadratic niche) responses relative to predictors. The default model.

```python
p_gaussian = compute_dark_diversity(
    y, x,
    model_type="gaussian",
    method="svi"
)
```

#### 3. Bayesian Neural Network Model (`model_type="bnn"`)

Models highly complex, non-linear interactions using a single hidden-layer BNN. Less interpretable, but suitable for complex datasets and mixed continuous/one-hot inputs.

```python
p_bnn = compute_dark_diversity(
    y, x,
    model_type="bnn",
    method="svi",
    hidden_size=10  # size of BNN hidden layer
)
```

---

### Handling Categorical & Label-Encoded Data

Columns with dtypes of `category`, `object`, `bool`, or `string` are **automatically auto-detected** and one-hot encoded, while continuous variables are standardized.

If your categorical data is **label-encoded as integers** (e.g. `landuse` represented by `0, 1, 2`), specify them explicitly using `categorical_cols` to prevent the model from treating them as continuous:

```python
predictions = compute_dark_diversity(
    y, x,
    model_type="linear",
    categorical_cols=["landuse"]
)
```

---

### Counterfactual Prediction Flow

To calculate dark diversity efficiently, we recommend using the object-oriented `PMFDark` model class. This allows you to fit the model once and retrieve all three outputs:

```python
from pmf_dark import PMFDark

model = PMFDark(model_type="gaussian", num_factors=2)
model.fit(y, x)

# 1. Full prediction (environment + latent factors)
# Represents the current distribution of the species with all drivers active
p_dist = model.distribution()

# 2. Counterfactual prediction (environment only)
# Represents the potential species pool (suitable habitat without unmeasured limitation drivers)
p_pool = model.pool()

# 3. Dark Diversity
# Represents the potential species pool value where a species is NOT observed (i.e. where y_obs == 0).
# Where a species is observed (y_obs > 0), the value is set to NA (NaN).
p_dark = model.dark()
```

For backward compatibility, you can also use `compute_dark_diversity` directly, although getting multiple outputs this way requires fitting the model multiple times:

```python
# 1. Full prediction (environment + latent factors)
p_full = compute_dark_diversity(y, x, include_latent=True)

# 2. Counterfactual prediction (environment only)
p_env = compute_dark_diversity(y, x, include_latent=False)
```

---

### Working with Count Data

If your species matrix `y` contains counts (integers $\ge 0$) instead of binary presence/absence, the package automatically infers the data type and fits a **Poisson** likelihood instead of Bernoulli:

```python
# y contains count values (e.g., abundance)
abundance_predictions = compute_dark_diversity(
    y_abundance, x,
    model_type="gaussian"
)
```

---

### Extra Evaluation & Plotting Utilities

The package includes utility modules under `extras/` to evaluate model performance and plot predictions:

```python
from extras.evaluation import compute_overall_error_metrics
from extras.plots import plot_environmental_response, plot_spatial_predictions

# 1. Evaluate performance (returns AUC, Brier Score, F1, etc.)
metrics = compute_overall_error_metrics(
    true_probabilities=true_values,
    predicted_probabilities=p_gaussian,
    observed_y=y
)
print("Model Performance:", metrics)

# 2. Plot spatial probability distribution maps
plot_spatial_predictions(
    probabilities=p_gaussian,
    coords=coords,
    species_name="species_1",
    y=y
)
```

---

### R Integration

If you prefer to work in R, you can use the R wrapper package **[pmf_dark_r](https://github.com/davidyshen/pmf_dark_r)**.

This wrapper allows you to run the Bayesian Probabilistic Matrix Factorisation model and estimate dark diversity directly inside your R environment (using `reticulate` under the hood to interface with this Python package).

For installation and usage instructions in R, check out the [pmf_dark_r repository](https://github.com/davidyshen/pmf_dark_r).

## Repository Structure

- `README.md` - This file
- `demo.ipynb` - Demonstration notebook
- `pyproject.toml` - Poetry/packaging configuration
- **data/**
  - `data/survey.csv` - Species presence/absence matrix (sites × species)
  - `data/env.csv` - Environmental predictors (sites × covariates)
  - `data/truth.csv` - Ground truth data (if available)
  - `data/pool_darkdiv.csv` - Simulated dark diversity pool
- **src/pmf_dark/** - Core package source code
  - `src/pmf_dark/__init__.py` - Package initialization and main imports
  - `src/pmf_dark/darkdiv.py` - Main dark diversity estimation entrypoint
  - `src/pmf_dark/inference.py` - SVI and MCMC inference execution
  - `src/pmf_dark/models.py` - Probabilistic response models (Linear, Gaussian, BNN)
- **extras/**
  - `extras/evaluation.py` - Performance evaluation and metrics
  - `extras/plots.py` - Plotting and visualization helper functions
- **tests/**
  - `tests/test_darkdiv.py` - Unit and integration tests

## Data Structure

The `compute_dark_diversity` function expects input pandas dataframes with specific column structures. Some sample data files are included in this repository for testing and demonstration (see demo.ipynb for usage).

### `survey.csv`

Species presence-absence matrix (sites × species), containing coordinates and ID columns.

```csv
"",ID,x,y,sp1,sp2,...,sp100
"1",1,3.5,3.5,0,0,...,0
"2",2,10.5,3.5,0,0,...,0
...
```

- **Row Index (`""`)**: Unnamed first column containing site labels/indices (loaded with `index_col=0`).
- **`ID`**: Site identifier column.
- **`x`, `y`**: Spatial coordinates of the sites.
- **`sp1` to `sp100`**: Binary species presence/absence markers (`1` for present, `0` for absent).

### `env.csv`

Environmental predictor matrix (sites × covariates) containing abiotic variables and land-use data.

```csv
"",ID,temperature,pH,elevation,landuse
"1",1,19.6969,5.9807,30.303,"1"
"2",2,19.6969,5.9583,101.01,"1"
...
```

- **Row Index (`""`)**: Unnamed first column matching the index labels in `survey.csv`.
- **`ID`**: Site identifier column.
- **`temperature`, `pH`, `elevation`**: Continuous environmental covariates.
- **`landuse`**: Categorical covariate (integer or string values, e.g., `"0"`, `"1"`).

### `truth.csv`

Simulated truth data to evaluate model performance, containing the true probabilities of species occurrence.

```csv
"",sp1,sp2,...,sp100
"1",0.05703,1.23e-05,...,0.00160
"2",0.07283,0.00012,...,0.00885
...
```

- **Row Index (`""`)**: Unnamed first column matching the site indices.
- **`sp1` to `sp100`**: Continuous probabilities of species occurrence (floats between 0 and 1).

### `pool_darkdiv.csv`

Dark diversity calculated using the pairwise co-occurrence method using the `DarkDiv` R package, included for comparison with our PMF-dark estimates.

```csv
"",sp1,sp2,...,sp100
"1",0.99999,0.35203,...,0.25145
"2",0.99999,0.50974,...,0.38711
...
```

- **Row Index (`""`)**: Unnamed first column matching the site indices.
- **`sp1` to `sp100`**: Continuous simulated occurrence probabilities for species.

## Advantages of This Approach

✓ **No subjective benchmarking**: Automated separation of environmental vs. unmeasured effects  
✓ **Mathematically principled**: Latent factors naturally absorb degradation signals  
✓ **Scalable**: SVI handles thousands of species and sites  
✓ **Species-specific**: Each species can have unique environmental responses  
✓ **Reproducible**: Fully probabilistic framework with clear assumptions

## Limitations

- Requires sufficient environmental variation to estimate effects reliably
- May overestimate dark diversity if detection is imperfect
- Computational cost increases with number of species and sites
- Requires careful tuning of number of latent factors

---

## Theoretical & Methodology Background

### The Problem: What is Dark Diversity?

Traditional biodiversity assessments only count observed species (alpha diversity). However, many species are absent from sites where they *could* thrive based on environmental conditions. This "**dark diversity**" represents:

- Species lost due to historical or ongoing land-use degradation
- Species unable to reach suitable sites due to dispersal limitation
- Species suppressed by biotic interactions

Quantifying dark diversity is crucial for:

- Conservation planning and restoration potential assessment
- Understanding true biodiversity patterns
- Identifying areas with highest restoration value

### Methodology

#### Core Model

The framework decomposes species occurrence probabilities into **three additive components**:

$$\text{logit}(p_{ij}) = \underbrace{\alpha_j}_{\text{Intercept}} + \underbrace{f_j(\mathbf{x}_i)}_{\text{Environmental Effects}} + \underbrace{\mathbf{w}_i^\top \mathbf{z}_j}_{\text{Latent Factors}}$$

Where:

- **$\alpha_j$**: Species-specific baseline prevalence
- **$f_j(\mathbf{x}_i)$**: Environmental response function to measured abiotic variables (temperature, pH, elevation, etc.), which can be modelled as linear, Gaussian niche, or non-linear (e.g. Bayesian neural network)
- **$\mathbf{w}_i^\top \mathbf{z}_j$**: Latent factors capturing unmeasured drivers of absence

#### Key Innovation: Counterfactual Predictions

1. **Full Predictions**: Include all components (environment + latent factors)
   - Represents observed diversity with all drivers active

2. **Environment-Only Predictions**: Exclude latent factors
   - Represents potential diversity (setting $\mathbf{w}_i^\top \mathbf{z}_j = 0$)

3. **Dark Diversity Proxy**: Difference between full and environment-only predictions
   - Quantifies species lost to unmeasured stressors

#### Inference: Stochastic Variational Inference (SVI)

The model is fit using **Pyro-based SVI**, which:

- Handles high-dimensional ecological matrices efficiently
- Treats inference as an optimisation problem (ELBO maximisation)
- Scales to thousands of sites and species
- Requires minimal computational resources

### Acknowledgements

This package is an output of the [SustainScapes](https://bio.au.dk/en/research/research-centres/sustainscapes) project at Aarhus University, funded by the Novo Nordisk Foundation (grant NNF20OC0059595).
This project was lead by David Shen, with contributions from Emilie S. Lissner, Signe Normand, all at Aarhus University, and Tom Johnson at the University of Sheffield.
