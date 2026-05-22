import matplotlib.pyplot as plt
import numpy as np


def plot_spatial_predictions(
    coords,
    arrays,
    species_idx,
    labels=None,
    y=None,
    presence_ax=None,
    cmap="Blues",
    point_size=200,
):

    n_plots = len(arrays)

    if labels is None:
        labels = [f"Array {i}" for i in range(n_plots)]

    if isinstance(presence_ax, int):
        presence_ax = [presence_ax]

    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(6 * n_plots, 5),
    )

    if n_plots == 1:
        axes = [axes]

    for i, (ax, arr, label) in enumerate(zip(axes, arrays, labels)):

        species = arr.columns[species_idx]

        values = arr.iloc[:, species_idx]

        sc = ax.scatter(
            coords.iloc[:, 0],
            coords.iloc[:, 1],
            c=values,
            cmap=cmap,
            s=point_size,
        )

        ax.set_aspect("equal")

        ax.set_title(f"{label} ({species})")

        fig.colorbar(sc, ax=ax)

        if y is not None and (presence_ax is None or i in presence_ax):

            mask = y.iloc[:, species_idx] == 1

            ax.scatter(
                coords.loc[mask, coords.columns[0]],
                coords.loc[mask, coords.columns[1]],
                marker="x",
                color="red",
                s=100,
                label="Observed presence",
            )

            ax.legend()

    plt.tight_layout()
    plt.show()



def plot_spatial_uncertainty(
    samples,
    species_idx,
    grid_shape=(15, 15),
    stats=("mean", "sd", "interval_width"),
    cmap_mean="Blues",
    cmap_uncertainty="magma",
    species_names=None,
):

    """
    Plot posterior mean and uncertainty for one species.

    samples shape:
    [num_samples, num_sites, num_species]
    """

    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()

    p_species = samples[:, :, species_idx]

    if species_names is not None:
        species_label = species_names[species_idx]
    else:
        species_label = f"Species {species_idx}"

    plot_arrays = []
    labels = []
    cmaps = []

    if "mean" in stats:

        plot_arrays.append(
            p_species.mean(axis=0).reshape(grid_shape)
        )

        labels.append("Mean")

        cmaps.append(cmap_mean)

    if "sd" in stats:

        plot_arrays.append(
            p_species.std(axis=0).reshape(grid_shape)
        )

        labels.append("Posterior SD")

        cmaps.append(cmap_uncertainty)

    if "interval_width" in stats:

        lower = np.quantile(
            p_species,
            0.025,
            axis=0,
        )

        upper = np.quantile(
            p_species,
            0.975,
            axis=0,
        )

        plot_arrays.append(
            (upper - lower).reshape(grid_shape)
        )

        labels.append("95% Interval Width")

        cmaps.append(cmap_uncertainty)

    n_plots = len(plot_arrays)

    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(5 * n_plots, 4),
    )

    if n_plots == 1:
        axes = [axes]

    for ax, arr, label, cmap in zip(
        axes,
        plot_arrays,
        labels,
        cmaps,
    ):

        im = ax.imshow(
            arr,
            origin="lower",
            cmap=cmap,
        )

        ax.set_aspect("equal")

        ax.set_title(
            f"{label} ({species_label})"
        )

        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()



def plot_environmental_response(
    x,
    predictions,
    species_idx,
    labels=None,
    colors=None,
    true_values=None,
    alpha=0.6,
):

    n_env = x.shape[1]

    if labels is None:
        labels = [
            f"Prediction {i}"
            for i in range(len(predictions))
        ]

    if colors is None:
        colors = [None] * len(predictions)

    species = predictions[0].columns[species_idx]

    plt.figure(figsize=(6 * n_env, 5))

    for j, col in enumerate(x.columns):

        plt.subplot(1, n_env, j + 1)

        for pred, label, color in zip(
            predictions,
            labels,
            colors,
        ):

            plt.scatter(
                x.iloc[:, j],
                pred.iloc[:, species_idx],
                alpha=alpha,
                color=color,
                label=label,
            )

        if true_values is not None:

            plt.scatter(
                x.iloc[:, j],
                true_values.iloc[:, species_idx],
                alpha=0.4,
                color="red",
                label="True values",
            )

        plt.xlabel(col)

        plt.ylabel("Prediction")

        plt.title(species)

        plt.legend(loc="upper left")

    plt.tight_layout()

    plt.show()