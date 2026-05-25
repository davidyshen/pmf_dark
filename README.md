# Bayesian Matrix Factorisation Community Modelling (BLFCM): Using matrix factorisation for dark diversity estimation

## Overview

This repository implements a **BMFCM** using matrix factorization to estimate **dark diversity**—the set of species absent from a site despite having suitable environmental conditions. The method uses **counterfactual predictions** to reconstruct the potential species pool by separating environmental effects from unmeasured drivers of absence (e.g., land-use degradation, dispersal limitation, biotic interactions).

## The Problem: What is Dark Diversity?

Traditional biodiversity assessments only count observed species (alpha diversity). However, many species are absent from sites where they *could* thrive based on environmental conditions. This "**dark diversity**" represents:

- Species lost due to historical or ongoing land-use degradation
- Species unable to reach suitable sites due to dispersal limitation
- Species suppressed by biotic interactions

Quantifying dark diversity is crucial for:
- Conservation planning and restoration potential assessment
- Understanding true biodiversity patterns
- Identifying areas with highest restoration value

## Methodology

### Core Model

The framework decomposes species occurrence probabilities into **three additive components**:

$$\text{logit}(p_{ij}) = \underbrace{\alpha_j}_{\text{Intercept}} + \underbrace{\mathbf{x}_i^\top \boldsymbol{\beta}_j}_{\text{Environmental Effects}} + \underbrace{\mathbf{w}_i^\top \mathbf{z}_j}_{\text{Latent Factors}}$$

Where:
- **$\alpha_j$**: Species-specific baseline prevalence
- **$\mathbf{x}_i^\top \boldsymbol{\beta}_j$**: Species response to measured abiotic variables (temperature, pH, elevation, etc.)
- **$\mathbf{w}_i^\top \mathbf{z}_j$**: Latent factors capturing unmeasured drivers of absence

### Key Innovation: Counterfactual Predictions

1. **Full Predictions**: Include all components (environment + latent factors)
   - Represents observed diversity with all drivers active
   
2. **Environment-Only Predictions**: Exclude latent factors
   - Represents potential diversity (setting $\mathbf{w}_i^\top \mathbf{z}_j = 0$)
   
3. **Dark Diversity Proxy**: Difference between full and environment-only predictions
   - Quantifies species lost to unmeasured stressors

### Inference: Stochastic Variational Inference (SVI)

The model is fit using **Pyro-based SVI**, which:
- Handles high-dimensional ecological matrices efficiently
- Treats inference as an optimization problem (ELBO maximization)
- Scales to thousands of sites and species
- Requires minimal computational resources

## Repository Structure

```
Matrix-factorisation/
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

## Installation

### Requirements

- Python 3.13 or 3.14
- PyTorch (with CUDA support if using GPU)
- Pyro (pyro-ppl)
- Pandas
- NumPy
- scikit-learn
- scipy

### Setup

To ensure PyTorch is installed with the correct CUDA version for your system, it is recommended to install PyTorch manually **first** before installing the package or its other dependencies.

#### 1. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/davidyshen/Matrix-factorisation.git
cd Matrix-factorisation

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### 2. Install PyTorch with CUDA
Visit the [PyTorch Getting Started guide](https://pytorch.org/get-started/locally/) to select the correct command for your CUDA version and OS. For example, to install PyTorch with CUDA 12.4 support on Windows/Linux:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

#### 3. Install remaining dependencies
If installing via pip:
```bash
pip install pyro-ppl pandas numpy scikit-learn scipy jupyter
```

If using Poetry:
```bash
# This will install the package and its remaining dependencies into your environment
poetry install
```

## Usage

### Running the Full Analysis

1. Prepare your data in `data/` directory:
   - `survey.csv`: Species presence/absence (rows = sites, columns = species, values = 0/1)
   - `env.csv`: Environmental predictors (rows = sites, columns = variables)

2. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook mat_fact_dark_div.ipynb
   ```

3. The notebook will:
   - Load and standardize data
   - Fit the matrix factorization model (2,500 iterations)
   - Generate predictions and save CSV outputs

### Customization

Key parameters in the notebook:

```python
# Model parameters
num_factors = 5                # Number of latent factors (adjust based on data complexity)
num_iterations = 2500          # Model iterations

# Learning rate
Adam({"lr": 0.01})            # Adjust if convergence is slow
```

## Output Files

- **mat_fact_predicted_probabilities_full.csv**: Predicted species occurrence probabilities including all effects
- **mat_fact_predicted_probabilities_env_only.csv**: Predicted probabilities using only environmental effects
- **mat_fact_dark_diversity_proxy.csv**: Dark diversity estimates (full - env_only)

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

## Interpretation of Results

### Dark Diversity Proxy Values
- **High values (close to 1)**: Species should be present based on environment but are absent—candidate for restoration
- **Low values (close to 0)**: Species absence explained by environmental conditions
- **Negative values**: Model predicts species should be absent (rare, indicates environmental unsuitability)

### Key Metrics
- **AUC (Area Under ROC Curve)**: Overall model discrimination (0.5 = random, 1.0 = perfect)
- **Brier Score**: Prediction calibration error (lower is better)
- **F1 Score**: Balance between precision and recall

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

## References & Theoretical Background

### Key Concepts
- **Joint Species Distribution Models (JSDMs)**: Latent variable models for multivariate species data
- **Matrix Factorization**: Low-rank decomposition of high-dimensional species matrices
- **Stochastic Variational Inference**: Scalable Bayesian inference for probabilistic models
- **Counterfactual Predictions**: Causal inference approach to estimate potential outcomes
