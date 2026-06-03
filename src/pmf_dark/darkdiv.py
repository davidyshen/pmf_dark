import torch
import pyro
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def infer_y_type(y):

    # torch tensor
    if isinstance(y, torch.Tensor):

        values = y.detach().cpu().numpy()

    # pandas dataframe
    elif hasattr(y, "to_numpy"):

        values = y.to_numpy()

    # numpy or other
    else:

        values = np.asarray(y)

    # Check missing values and data type validity
    try:
        if pd.isna(values).any():
            raise ValueError("y contains missing values.")

        # Binary data
        if np.isin(values, [0, 1]).all():
            return "presence_absence"

        # Count data
        elif (values >= 0).all():
            return "count"

        else:
            raise ValueError("y must contain either binary or count data.")
    except TypeError:
        raise ValueError("y must contain numeric (binary or count) data.")


def prepare_data(x, y, categorical_cols=None, cuda=False):

    # Keep names
    y_columns = y.columns
    site_index = y.index

    if categorical_cols is None:
        categorical_cols = []
    else:
        categorical_cols = list(categorical_cols)

    # Validate that all specified categorical_cols exist in x
    missing_cols = [col for col in categorical_cols if col not in x.columns]
    if missing_cols:
        raise ValueError(
            f"The following specified categorical columns are not in x: {missing_cols}"
        )

    # Auto-detect categorical columns based on dtypes (category, object, bool, string)
    auto_cat_cols = []
    for col in x.columns:
        if col not in categorical_cols:
            if isinstance(x[col].dtype, pd.CategoricalDtype) or x[col].dtype in [
                "object",
                "bool",
                "string",
            ]:
                auto_cat_cols.append(col)

    all_cat_cols = categorical_cols + auto_cat_cols

    # Process x
    if len(all_cat_cols) > 0:
        x_temp = x.copy()
        # Convert all to category type so pd.get_dummies knows they are categorical
        for col in all_cat_cols:
            x_temp[col] = x_temp[col].astype("category")

        # Get continuous columns
        cont_cols = [col for col in x.columns if col not in all_cat_cols]

        # Standardise continuous columns
        if len(cont_cols) > 0:
            x_cont = x_temp[cont_cols]
            # Handle standard deviation of 0
            x_std = x_cont.std().replace(0, 1.0)
            x_cont_std = (x_cont - x_cont.mean()) / x_std
        else:
            x_cont_std = pd.DataFrame(index=site_index)

        # One-hot encode categorical columns
        x_dummies = pd.get_dummies(
            x_temp[all_cat_cols],
            prefix=all_cat_cols,
            prefix_sep="_",
            drop_first=False,
            dtype=float,
        )

        # Concatenate standardized continuous columns and dummy columns
        x_processed = pd.concat([x_cont_std, x_dummies], axis=1)
    else:
        # Standardise all columns (previous behaviour)
        x_std = x.std().replace(0, 1.0)
        x_processed = (x - x.mean()) / x_std

    x_columns = x_processed.columns

    # Convert to tensors
    x_tensor = torch.tensor(
        x_processed.to_numpy(),
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y.to_numpy(),
        dtype=torch.float32,
    )

    # Device
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)

    return {
        "x": x_tensor,
        "y": y_tensor,
        "x_columns": x_columns,
        "y_columns": y_columns,
        "site_index": site_index,
    }


def compute_predictions(
    samples, x, model_type="gaussian", include_latent=True, y_type="presence_absence"
):

    if model_type == "linear":
        alpha = samples["alpha"].squeeze(1)
        beta = samples["beta"].squeeze(1)
        eta = alpha[:, None, :] + torch.einsum("ij,sjk->sik", x, beta)

    elif model_type == "gaussian":
        alpha = samples["alpha"].squeeze(1)
        mu = samples["mu"].squeeze(1)
        gamma = samples["gamma"].squeeze(1)

        # x:     [sites, env]
        # mu:    [samples, env, species]
        # gamma: [samples, env, species]
        diff = x[None, :, :, None] - mu[:, None, :, :]
        env_effect = -torch.sum(gamma[:, None, :, :] * diff**2, dim=2)

        eta = alpha[:, None, :] + env_effect

    elif model_type == "bnn":
        w1 = samples["w1"].squeeze(1)
        b1 = samples["b1"].squeeze(1)
        w2 = samples["w2"].squeeze(1)
        b2 = samples["b2"].squeeze(1)

        hidden = torch.tanh(torch.einsum("ij,sjh->sih", x, w1) + b1[:, None, :])

        eta = torch.einsum("sih,shk->sik", hidden, w2) + b2[:, None, :]

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if include_latent:
        W = samples["W"].squeeze(1)
        Z = samples["Z"].squeeze(1)
        eta = eta + torch.einsum("sik,sjk->sij", W, Z)

    if y_type == "presence_absence":
        return torch.sigmoid(eta)

    elif y_type == "count":
        return torch.exp(eta)

    else:
        raise ValueError("y_type must be 'presence_absence' or 'count'")


class PMFDark:
    """
    Object-oriented API for PMF (Bayesian Probabilistic Matrix Factorisation
    for Dark Diversity) model.

    This class separates the computationally heavy model fitting stage from the
    prediction extraction stage. It allows you to train the response model and
    the latent factors once, and then quickly generate current distribution,
    potential species pool, and dark diversity predictions without refitting.

    Attributes:
        model_type (str): Ecological response model type ("linear", "gaussian", "bnn").
        num_factors (int): Number of latent factors for species residual covariance.
        method (str): Inference method ("svi" or "mcmc").
        cuda (bool): Whether to use GPU for SVI.
        init_kwargs (dict): Additional response model configurations.
        is_fitted (bool): True if the model has been fitted successfully.
        fit_result (dict): Inference fitting output containing guide, losses, and samples.
        y_type (str): Type of species data ("presence_absence" or "count").
        y_train (pandas.DataFrame or array-like): Original species presence/absence or count matrix.
        x_train_tensor (torch.Tensor): Processed environment training tensor.
        y_train_tensor (torch.Tensor): Processed species training tensor.
        x_columns (Index): Processed environment feature/column names.
        y_columns (Index): Species column names.
        site_index (Index): Site indices/names.
    """

    def __init__(
        self,
        model_type="gaussian",
        num_factors=1,
        method="svi",
        cuda=False,
        **kwargs,
    ):
        """
        Initialize the PMFDark model architecture and inference settings.

        Args:
            model_type (str): Response model to model species responses. Options:
                - "linear": Linear response on the logit scale.
                - "gaussian": Quadratic niche response (default).
                - "bnn": Bayesian Neural Network with a single hidden layer.
            num_factors (int): Number of latent factors used to capture residual
                species covariance (default: 1).
            method (str): Inference method. Options:
                - "svi": Stochastic Variational Inference (default).
                - "mcmc": Hamiltonian NUTS MCMC.
            cuda (bool): If True, uses GPU acceleration for SVI training.
            **kwargs: Extra model specific configurations (e.g., hidden_size=10 for BNN).
        """
        self.model_type = model_type
        self.num_factors = num_factors
        self.method = method
        self.cuda = cuda
        self.init_kwargs = kwargs

        self.is_fitted = False
        self.fit_result = None
        self.y_type = None

        # Training data attributes stored after fitting
        self.y_train = None
        self.x_train_tensor = None
        self.y_train_tensor = None
        self.x_columns = None
        self.y_columns = None
        self.site_index = None

    def fit(self, y, x, categorical_cols=None, batch_size=None, **kwargs):
        """
        Fit the PMF model to species matrix y and environmental predictors x.

        Args:
            y (pandas.DataFrame or array-like): Species presence-absence (binary) or
                counts matrix of shape (n_sites, n_species).
            x (pandas.DataFrame or array-like): Environmental predictor matrix of
                shape (n_sites, n_env).
            categorical_cols (list, optional): Explicit list of column names in x
                to treat as categorical variables.
            batch_size (int, optional): Mini-batch size for SVI training. Defaults to None.
            **kwargs: Additional hyperparameters for SVI training (e.g., lr,
                num_iterations, num_samples) or MCMC (e.g., num_samples, warmup_steps, num_chains).

        Returns:
            PMFDark: The fitted model instance.

        Raises:
            ValueError: If the combination of method and model_type is invalid.
        """
        if self.cuda and not torch.cuda.is_available():
            import warnings

            warnings.warn(
                "CUDA was requested (cuda=True), but PyTorch cannot detect a CUDA-enabled GPU. "
                "Falling back to CPU."
            )

        if self.method == "mcmc" and self.model_type == "bnn":
            raise ValueError(
                "MCMC is currently not supported for the Bayesian neural network model. "
                "Please use method='svi' instead."
            )

        # Check if y is presence/absence or count data
        self.y_type = infer_y_type(y)

        # Preprocess and prepare data
        data = prepare_data(x, y, categorical_cols=categorical_cols, cuda=self.cuda)

        self.y_train = y
        self.x_train_tensor = data["x"]
        self.y_train_tensor = data["y"]
        self.x_columns = data["x_columns"]
        self.y_columns = data["y_columns"]
        self.site_index = data["site_index"]

        # Load the model definition function
        if self.model_type == "linear":
            from .models import linear_model

            model_fn = linear_model
        elif self.model_type == "gaussian":
            from .models import gaussian

            model_fn = gaussian
        elif self.model_type == "bnn":
            from .models import bnn_model

            model_fn = bnn_model
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Combine initialization and fit parameters
        fit_kwargs = {**self.init_kwargs, **kwargs}

        # Run inference
        if self.method == "svi":
            from .inference import fit_svi

            self.fit_result = fit_svi(
                model_fn,
                self.y_train_tensor,
                self.x_train_tensor,
                self.num_factors,
                y_type=self.y_type,
                cuda=self.cuda,
                batch_size=batch_size,
                **fit_kwargs,
            )
        elif self.method == "mcmc":
            from .inference import fit_mcmc

            self.fit_result = fit_mcmc(
                model_fn,
                self.y_train_tensor,
                self.x_train_tensor,
                self.num_factors,
                y_type=self.y_type,
                batch_size=batch_size,
                **fit_kwargs,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted = True
        return self

    def _predict(
        self,
        include_latent=True,
        pred_batch_size=None,
        return_means=True,
    ):
        """
        Internal method to generate species predictions from the fitted model.

        Args:
            include_latent (bool): If True, includes latent factors in predictions.
                If False, generates counterfactual (environment-only) predictions.
            pred_batch_size (int, optional): Site-chunk size for chunked predictions.
            return_means (bool): If True, returns posterior means. If False, returns
                full posterior samples.

        Returns:
            pandas.DataFrame or numpy.ndarray: Species presence probability or count predictions.

        Raises:
            RuntimeError: If called before fitting the model.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before computing predictions. Call .fit() first."
            )

        fit = self.fit_result
        x = self.x_train_tensor
        model_type = self.model_type
        y_type = self.y_type
        site_index = self.site_index
        y_columns = self.y_columns

        samples = fit["samples"]

        # Compute probabilities
        if pred_batch_size is not None:
            n_sites = x.shape[0]
            pred_chunks = []
            for i in range(0, n_sites, pred_batch_size):
                x_chunk = x[i : i + pred_batch_size]
                samples_chunk = samples.copy()
                if "W" in samples:
                    w_tensor = samples["W"]
                    if w_tensor.dim() == 4:
                        samples_chunk["W"] = w_tensor[:, :, i : i + pred_batch_size, :]
                    else:
                        samples_chunk["W"] = w_tensor[:, i : i + pred_batch_size, :]

                pred_chunk = compute_predictions(
                    samples_chunk,
                    x_chunk,
                    model_type=model_type,
                    include_latent=include_latent,
                    y_type=y_type,
                )

                if return_means:
                    pred_chunk_processed = pred_chunk.mean(dim=0).detach().cpu().numpy()
                else:
                    pred_chunk_processed = pred_chunk.detach().cpu().numpy()

                pred_chunks.append(pred_chunk_processed)

            if return_means:
                pred_np = np.concatenate(pred_chunks, axis=0)
                pred = pd.DataFrame(
                    pred_np,
                    index=site_index,
                    columns=y_columns,
                )
            else:
                # If returning full samples, shape of each chunk is (num_samples, batch_size, n_species)
                # We concatenate along the site dimension (axis 1)
                pred = np.concatenate(pred_chunks, axis=1)

        else:
            pred = compute_predictions(
                samples,
                x,
                model_type=model_type,
                include_latent=include_latent,
                y_type=y_type,
            )
            if return_means:
                pred = pred.mean(dim=0)
                pred = pred.detach().cpu().numpy()

                pred = pd.DataFrame(
                    pred,
                    index=site_index,
                    columns=y_columns,
                )

            else:
                pred = pred.detach().cpu().numpy()

        return pred

    def distribution(self, pred_batch_size=None, return_means=True):
        """
        Generate species occurrence predictions including latent factors.

        Represents the current distribution of the species with all drivers active.

        Args:
            pred_batch_size (int, optional): Site-chunk size for prediction output.
            return_means (bool): If True, returns a pandas.DataFrame of posterior means.
                If False, returns a NumPy array of raw posterior samples.

        Returns:
            pandas.DataFrame or numpy.ndarray: Presence probability or count predictions.
        """
        return self._predict(
            include_latent=True,
            pred_batch_size=pred_batch_size,
            return_means=return_means,
        )

    def pool(self, pred_batch_size=None, return_means=True):
        """
        Generate counterfactual environment-only predictions (excluding latent factors).

        Represents the potential species pool (suitable habitat if the unmeasured
        limitation drivers/stressors were not there).

        Args:
            pred_batch_size (int, optional): Site-chunk size for prediction output.
            return_means (bool): If True, returns a pandas.DataFrame of posterior means.
                If False, returns a NumPy array of raw posterior samples.

        Returns:
            pandas.DataFrame or numpy.ndarray: Counterfactual presence probability or count predictions.
        """
        return self._predict(
            include_latent=False,
            pred_batch_size=pred_batch_size,
            return_means=return_means,
        )

    def dark(self, pred_batch_size=None, return_means=True):
        """
        Generate predictions of species dark diversity.

        Defined as the potential species pool value (.pool()) where a species is not
        observed in the original training data (i.e. where y == 0). Where a species
        is observed in the original training data, the dark diversity is set to NA (NaN).

        Args:
            pred_batch_size (int, optional): Site-chunk size for prediction output.
            return_means (bool): If True, returns a pandas.DataFrame of posterior means.
                If False, returns a NumPy array of raw posterior samples.

        Returns:
            pandas.DataFrame or numpy.ndarray: Dark diversity predictions with observed entries
                masked to NaN.
        """
        pool_pred = self.pool(
            pred_batch_size=pred_batch_size,
            return_means=return_means,
        )

        if return_means:
            # pool_pred is a pandas DataFrame.
            # Mask out observed entries by setting them to np.nan.
            # We want to keep values where self.y_train == 0.
            return pool_pred.where(self.y_train == 0, np.nan)
        else:
            # pool_pred is a numpy array of shape (num_samples, n_sites, n_species).
            # self.y_train is a pandas DataFrame of shape (n_sites, n_species).
            observed_mask = self.y_train.to_numpy() > 0

            # We copy pool_pred and set the observed elements to np.nan across all samples.
            dark_pred = np.copy(pool_pred)
            dark_pred[:, observed_mask] = np.nan
            return dark_pred


def compute_dark_diversity(
    y,
    x,
    model_type="gaussian",
    num_factors=1,
    method="svi",
    cuda=False,
    include_latent=True,
    return_means=True,
    batch_size=None,
    pred_batch_size=None,
    categorical_cols=None,
    **kwargs,
):
    """
    Fit a PMF model and compute species occurrence or pool predictions.

    This function wraps the object-oriented PMFDark class to ensure 100% backward
    compatibility with earlier functional usage.

    Args:
        y (pandas.DataFrame or array-like): Species presence-absence or counts matrix.
        x (pandas.DataFrame or array-like): Environmental predictor matrix.
        model_type (str, default "gaussian"): Response model type: "linear" | "gaussian" | "bnn".
        num_factors (int, default 1): Number of latent factors.
        method (str, default "svi"): Inference method: "svi" | "mcmc".
        cuda (bool, default False): Use GPU computation (SVI only).
        include_latent (bool, default True): Include latent factors in predictions.
        return_means (bool, default True): Return posterior means (True) or raw samples (False).
        batch_size (int, optional): Mini-batch size for training.
        pred_batch_size (int, optional): Site-chunk size for predictions.
        categorical_cols (list, optional): Explicit list of columns to treat as categorical.
        **kwargs: Extra model/method specific arguments (e.g. lr, num_iterations, num_samples).

    Returns:
        pandas.DataFrame or numpy.ndarray: Probability/count predictions.
    """
    model = PMFDark(
        model_type=model_type,
        num_factors=num_factors,
        method=method,
        cuda=cuda,
        **kwargs,
    )

    model.fit(
        y=y,
        x=x,
        categorical_cols=categorical_cols,
        batch_size=batch_size,
    )

    if include_latent:
        return model.distribution(
            pred_batch_size=pred_batch_size,
            return_means=return_means,
        )
    else:
        return model.pool(
            pred_batch_size=pred_batch_size,
            return_means=return_means,
        )
