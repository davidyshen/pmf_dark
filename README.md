# PMF-dark: Using matrix factorisation for dark diversity estimation

## Overview

This repository implements a **PMF-dark** using Bayesian Probabilistic Matrix Factorisation to estimate **dark diversity** - the set of species absent from a site despite having suitable environmental conditions. The method uses **counterfactual predictions** to reconstruct the potential species pool by separating environmental effects from unmeasured drivers of absence (e.g., land-use degradation, dispersal limitation, biotic interactions).

## Installation

### Requirements

- Python 3.13 or 3.14
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

### Quick Start: Basic API Usage

```python
import pandas as pd
from pmf_dark import compute_dark_diversity

# 1. Load data
y = pd.read_csv("data/survey.csv", index_col=0)
x = pd.read_csv("data/env.csv", index_col=0)

# Drop non-species/non-environmental metadata
coords = y[["x", "y"]]
y = y.drop(columns=["x", "y", "ID"])
x = x.drop(columns=["ID"])

# 2. Run Dark Diversity Estimation using a Gaussian Niche Model with SVI
predictions = compute_dark_diversity(
    y=y,
    x=x,
    model_type="gaussian",   # Ecological response model
    method="svi",            # Stochastic Variational Inference
    num_factors=2,           # Latent factors
    num_iterations=2500,     # SVI parameters
    categorical_cols=["landuse"] # Explicitly treat landuse as categorical
)
```

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

To calculate dark diversity, run predictions both with and without latent factors:

```python
# 1. Full prediction (environment + latent factors)
p_full = compute_dark_diversity(
    y, x, model_type="gaussian", include_latent=True
)

# 2. Counterfactual prediction (environment only)
p_env = compute_dark_diversity(
    y, x, model_type="gaussian", include_latent=False
)

# 3. Dark Diversity Proxy (Species pool index)
dark_diversity = p_full - p_env
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

```
PMF_dark/
├── README.md                                    # This file
├── mat_fact_dark_div.ipynb                     # Main analysis notebook
├── data/
│   ├── survey.csv                              # Species presence/absence matrix (sites × species)
│   ├── env.csv                                 # Environmental predictors (sites × covariates)
│   └── truth.csv                               # Ground truth data (if available)
└── output/
    ├── mat_fact_predicted_probabilities_full.csv           # Full model predictions
    ├── mat_fact_predicted_probabilities_env_only.csv       # Environment-only predictions
    └── mat_fact_dark_diversity_proxy.csv                   # Dark diversity estimates
```

## Data Format

### survey.csv

```
site_id,species_1,species_2,...,species_n,ID,x,y
site_1,0,1,0,...,1,id_1,100.5,200.3
site_2,1,0,1,...,0,id_2,101.2,201.5
...
```

- Rows: Sites/locations
- Columns: Species (0/1 presence/absence) + ID + spatial coordinates
- **Note**: ID and spatial coordinates are automatically extracted/dropped

### env.csv

```
site_id,temp,pH,elevation,...,ID,landuse
site_1,15.2,7.1,500,...,id_1,degraded
site_2,14.8,6.9,520,...,id_2,pristine
...
```

- Rows: Sites matching survey.csv
- Columns: Environmental predictors + ID + land-use
- **Note**: ID and land-use columns are dropped; only abiotic predictors are used

## Output Files

- **mat_fact_predicted_probabilities_full.csv**: Predicted species occurrence probabilities including all effects
- **mat_fact_predicted_probabilities_env_only.csv**: Predicted probabilities using only environmental effects
- **mat_fact_dark_diversity_proxy.csv**: Dark diversity estimates (full - env_only)

## Advantages of This Approach

✓ **No subjective benchmarking**: Automated separation of environmental vs. unmeasured effects  
✓ **Mathematically principled**: Latent factors naturally absorb degradation signals  
✓ **Scalable**: SVI handles thousands of species and sites  
✓ **Species-specific**: Each species can have unique environmental responses  
✓ **Reproducible**: Fully probabilistic framework with clear assumptions

## Limitations

- Assumes species responses are log-linear (logit link)
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

### Interpretation of Results

#### Dark Diversity Proxy Values

- **High values (close to 1)**: Species should be present based on environment but are absent—candidate for restoration
- **Low values (close to 0)**: Species absence explained by environmental conditions
- **Negative values**: Model predicts species should be absent (rare, indicates environmental unsuitability)

#### Key Metrics

- **AUC (Area Under ROC Curve)**: Overall model discrimination (0.5 = random, 1.0 = perfect)
- **Brier Score**: Prediction calibration error (lower is better)
- **F1 Score**: Balance between precision and recall

### References & Literature

- **Joint Species Distribution Models (JSDMs)**: Latent variable models for multivariate species data
- **Matrix Factorisation**: Low-rank decomposition of high-dimensional species matrices
- **Stochastic Variational Inference**: Scalable Bayesian inference for probabilistic models
- **Counterfactual Predictions**: Causal inference approach to estimate potential outcomes
