import numpy as np
import pyro
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from torch import device
import torch
from pyro.infer import Predictive


def fit_svi(
    model,
    y,
    x,
    num_factors,
    y_type,
    lr=0.01,
    num_iterations=2500,
    num_samples=1000,
    cuda=False,
    batch_size=None,
    **kwargs,
):

    num_factors = int(num_factors)
    num_iterations = int(num_iterations)
    num_samples = int(num_samples)
    if batch_size is not None:
        batch_size = int(batch_size)

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define create_plates helper for autoguide subsampling
    def create_plates(*args, **kwargs):
        # args[1] is the species presence/absence matrix Y
        y_data = args[1]
        n_sites = y_data.shape[0]
        batch_size_val = kwargs.get("batch_size", None)
        return pyro.plate("sites", n_sites, subsample_size=batch_size_val)

    # Setup Inference
    pyro.clear_param_store()
    guide = AutoNormal(model, create_plates=create_plates)
    optimizer = Adam({"lr": lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Training Loop
    losses = []
    for i in range(num_iterations):
        loss = svi.step(x, y, num_factors, y_type, batch_size=batch_size, **kwargs)
        losses.append(loss)
        if i % 500 == 0:
            print(f"Iteration {i} - Loss: {loss:.2f}")

    # Simple convergence check
    first = np.mean(losses[-200:-100])
    last = np.mean(losses[-100:])
    relative_change = abs(last - first) / abs(first)
    if relative_change < 0.01:
        print("SVI converged successfully.")
    else:
        print("SVI may not have converged.")

    # return samples by running Predictive only on the guide
    # (running on the model would execute the likelihood forward pass and create a huge, unused "obs" tensor)
    predictive = Predictive(
        guide,
        num_samples=num_samples,
    )

    samples = predictive(x, y, num_factors, y_type, batch_size=None, **kwargs)

    return {
        "method": "svi",
        "guide": guide,
        "losses": losses,
        "samples": samples,
    }


def fit_mcmc(
    model,
    y,
    x,
    num_factors,
    y_type,
    num_samples=1000,
    warmup_steps=500,
    num_chains=1,
    batch_size=None,
    **kwargs,
):
    num_factors = int(num_factors)
    num_samples = int(num_samples)
    warmup_steps = int(warmup_steps)
    num_chains = int(num_chains)
    if batch_size is not None:
        batch_size = int(batch_size)

    if batch_size is not None:
        raise ValueError(
            "MCMC does not support mini-batching. Please use SVI (method='svi') or set batch_size=None."
        )
    pyro.clear_param_store()

    kernel = NUTS(model)

    mcmc = MCMC(
        kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
    )

    mcmc.run(x, y, num_factors, y_type)

    return {
        "method": "mcmc",
        "mcmc": mcmc,
        "samples": mcmc.get_samples(),
    }
