# 2. Define the Model
import pyro
import pyro.distributions as dist
import torch


def observation_dist(eta, y_type):
    if y_type == "presence_absence":
        return dist.Bernoulli(logits=eta)

    elif y_type == "count":
        return dist.Poisson(rate=torch.exp(torch.clamp(eta, -20, 20)))
    # elif y_type == "count":
    #
    #    mu = torch.exp(torch.clamp(eta, -20, 20))
    #
    #    dispersion = pyro.sample(
    #        "dispersion",
    #        dist.LogNormal(
    #            torch.zeros(eta.shape[-1], device=eta.device),
    #            torch.ones(eta.shape[-1], device=eta.device),
    #        ).to_event(1),
    #    )
    #
    #    return dist.NegativeBinomial(
    #        total_count=dispersion,
    #        logits=torch.log(mu) - torch.log(dispersion),
    #    )

    else:
        raise ValueError(
            "y_type must be 'presence_absence', 'count', or 'zero_inflated_count'"
        )


def linear_model(X, Y, num_factors, y_type="presence_absence", batch_size=None):
    num_factors = int(num_factors)
    if batch_size is not None:
        batch_size = int(batch_size)

    device = X.device

    n_sites, n_species = Y.shape
    n_env = X.shape[1]

    alpha = pyro.sample(
        "alpha",
        dist.Normal(
            torch.zeros(n_species, device=device),
            torch.ones(n_species, device=device),
        ).to_event(1),
    )

    beta = pyro.sample(
        "beta",
        dist.Normal(
            torch.zeros(n_env, n_species, device=device),
            torch.ones(n_env, n_species, device=device),
        ).to_event(2),
    )

    Z = pyro.sample(
        "Z",
        dist.Normal(
            torch.zeros(n_species, num_factors, device=device),
            torch.ones(n_species, num_factors, device=device),
        ).to_event(2),
    )

    with pyro.plate("sites", n_sites, subsample_size=batch_size) as ind:
        X_batch = X[ind]
        Y_batch = Y[ind]

        W_batch = pyro.sample(
            "W",
            dist.Normal(
                torch.zeros(num_factors, device=device),
                torch.ones(num_factors, device=device),
            ).to_event(1),
        )

        eta_batch = (
            alpha
            + torch.matmul(X_batch, beta)
            + torch.matmul(W_batch, Z.transpose(-1, -2))
        )

        pyro.sample(
            "obs",
            observation_dist(eta_batch, y_type).to_event(1),
            obs=Y_batch,
        )


def gaussian(
    X, Y, num_factors, y_type="presence_absence", batch_size=None
):
    num_factors = int(num_factors)
    if batch_size is not None:
        batch_size = int(batch_size)

    device = X.device

    n_sites, n_species = Y.shape
    n_env = X.shape[1]

    alpha = pyro.sample(
        "alpha",
        dist.Normal(
            torch.zeros(n_species, device=device),
            torch.ones(n_species, device=device),
        ).to_event(1),
    )

    # species-specific environmental optimum
    mu = pyro.sample(
        "mu",
        dist.Normal(
            torch.zeros(n_env, n_species, device=device),
            torch.ones(n_env, n_species, device=device),
        ).to_event(2),
    )

    # positive niche strength / inverse width
    gamma = pyro.sample(
        "gamma",
        dist.LogNormal(
            torch.zeros(n_env, n_species, device=device),
            torch.ones(n_env, n_species, device=device),
        ).to_event(2),
    )

    Z = pyro.sample(
        "Z",
        dist.Normal(
            torch.zeros(n_species, num_factors, device=device),
            torch.ones(n_species, num_factors, device=device),
        ).to_event(2),
    )

    with pyro.plate("sites", n_sites, subsample_size=batch_size) as ind:
        X_batch = X[ind]
        Y_batch = Y[ind]

        W_batch = pyro.sample(
            "W",
            dist.Normal(
                torch.zeros(num_factors, device=device),
                torch.ones(num_factors, device=device),
            ).to_event(1),
        )

        # quadratic environmental response
        diff = X_batch[..., None] - mu[..., None, :, :]
        env_effect = -torch.sum(gamma[..., None, :, :] * diff**2, dim=-2)

        eta_batch = alpha + env_effect + torch.matmul(W_batch, Z.transpose(-1, -2))

        pyro.sample(
            "obs",
            observation_dist(eta_batch, y_type).to_event(1),
            obs=Y_batch,
        )


def bnn_model(
    X, Y, num_factors=1, y_type="presence_absence", hidden_size=10, batch_size=None
):
    num_factors = int(num_factors)
    hidden_size = int(hidden_size)
    if batch_size is not None:
        batch_size = int(batch_size)

    device = X.device

    n_sites, n_species = Y.shape
    n_env = X.shape[1]

    w1 = pyro.sample(
        "w1",
        dist.Normal(
            torch.zeros(n_env, hidden_size, device=device),
            torch.ones(n_env, hidden_size, device=device),
        ).to_event(2),
    )

    b1 = pyro.sample(
        "b1",
        dist.Normal(
            torch.zeros(hidden_size, device=device),
            torch.ones(hidden_size, device=device),
        ).to_event(1),
    )

    w2 = pyro.sample(
        "w2",
        dist.Normal(
            torch.zeros(hidden_size, n_species, device=device),
            torch.ones(hidden_size, n_species, device=device),
        ).to_event(2),
    )

    b2 = pyro.sample(
        "b2",
        dist.Normal(
            torch.zeros(n_species, device=device),
            torch.ones(n_species, device=device),
        ).to_event(1),
    )

    Z = None
    if num_factors > 0:
        Z = pyro.sample(
            "Z",
            dist.Normal(
                torch.zeros(n_species, num_factors, device=device),
                torch.ones(n_species, num_factors, device=device),
            ).to_event(2),
        )

    with pyro.plate("sites", n_sites, subsample_size=batch_size) as ind:
        X_batch = X[ind]
        Y_batch = Y[ind]

        hidden = torch.tanh(X_batch @ w1 + b1)
        eta_batch = hidden @ w2 + b2

        if num_factors > 0:
            W_batch = pyro.sample(
                "W",
                dist.Normal(
                    torch.zeros(num_factors, device=device),
                    torch.ones(num_factors, device=device),
                ).to_event(1),
            )
            eta_batch = eta_batch + W_batch @ Z.transpose(-1, -2)

        pyro.sample(
            "obs",
            observation_dist(eta_batch, y_type).to_event(1),
            obs=Y_batch,
        )
