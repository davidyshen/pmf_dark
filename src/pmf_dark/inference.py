import numpy as np
import pyro
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from torch import device
import torch    
from pyro.infer import Predictive

def fit_svi(model, y, x, num_factors, y_type, lr=0.01, num_iterations=2500, cuda=False, **kwargs,):

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup Inference
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model)
    optimizer = Adam({"lr": lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Training Loop
    losses = []
    for i in range(num_iterations):
        loss = svi.step(x, y, num_factors, y_type, **kwargs)
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

    # return samples
    predictive = Predictive(
        model,
        guide=guide,
        num_samples=1000,
    )

    samples = predictive(x, y, num_factors, y_type, **kwargs)

    return {
        "method": "svi",
        "guide": guide,
        "losses": losses,
        "samples": samples,
    }

def fit_mcmc(model, y, x, num_factors, y_type, num_samples=1000, warmup_steps=500, num_chains=1):
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