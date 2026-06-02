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

    # Check missing values
    if np.isnan(values).any():
        raise ValueError("y contains missing values.")

    # Binary data
    if np.isin(values, [0, 1]).all():
        return "presence_absence"

    # Count data
    elif (values >= 0).all():
        return "count"

    else:
        raise ValueError("y must contain either binary or count data.")


def prepare_data(x, y, cuda=False):

    # Keep names
    x_columns = x.columns
    y_columns = y.columns
    site_index = y.index

    # Standardize x
    x = (x - x.mean()) / x.std()

    # Convert to tensors
    x_tensor = torch.tensor(
        x.to_numpy(),
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

    elif model_type in ["gaussian", "gaussian_response_model"]:
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
    **kwargs,
):

    if cuda and not torch.cuda.is_available():
        import warnings

        warnings.warn(
            "CUDA was requested (cuda=True), but PyTorch cannot detect a CUDA-enabled GPU. "
            "Please check your NVIDIA drivers and reinstall PyTorch with CUDA support. "
            "Falling back to CPU."
        )

    # For bnn only svi is supported
    if method == "mcmc" and model_type == "bnn":
        raise ValueError(
            "MCMC is currently not supported for the Bayesian neural network model. "
            "Please use method='svi' instead."
        )

    # Check if y is presence/absence or count data
    y_type = infer_y_type(y)
    print(y_type)

    data = prepare_data(x, y, cuda=cuda)

    x = data["x"]
    y = data["y"]

    # Load the model
    if model_type == "linear":
        from .models import linear_model

        model = linear_model
    elif model_type in ["gaussian", "gaussian_response_model"]:
        from .models import gaussian

        model = gaussian
    elif model_type == "bnn":
        from .models import bnn_model

        model = bnn_model

    # Inference
    if method == "svi":
        from .inference import fit_svi

        fit = fit_svi(
            model,
            y,
            x,
            num_factors,
            y_type=y_type,
            cuda=cuda,
            batch_size=batch_size,
            **kwargs,
        )
    elif method == "mcmc":
        from .inference import fit_mcmc

        fit = fit_mcmc(
            model, y, x, num_factors, y_type=y_type, batch_size=batch_size, **kwargs
        )

    # Compute probabilities
    if pred_batch_size is not None:
        n_sites = x.shape[0]
        pred_chunks = []
        for i in range(0, n_sites, pred_batch_size):
            x_chunk = x[i : i + pred_batch_size]
            samples_chunk = fit["samples"].copy()
            if "W" in fit["samples"]:
                w_tensor = fit["samples"]["W"]
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
                index=data["site_index"],
                columns=data["y_columns"],
            )
        else:
            # If returning full samples, shape of each chunk is (num_samples, batch_size, n_species)
            # We concatenate along the site dimension (axis 1)
            pred = np.concatenate(pred_chunks, axis=1)

    else:
        pred = compute_predictions(
            fit["samples"],
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
                index=data["site_index"],
                columns=data["y_columns"],
            )

        else:
            pred = pred.detach().cpu().numpy()

    return pred
