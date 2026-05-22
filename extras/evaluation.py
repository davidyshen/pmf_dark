import torch
import pandas as pd
import matplotlib.pyplot as plt


def to_tensor(x):

    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    return x


def compute_overall_error_metrics(predictions, true_values):

    predictions = to_tensor(predictions)
    true_values = to_tensor(true_values)

    mse = torch.mean((predictions - true_values) ** 2)

    rmse = torch.sqrt(mse)

    mae = torch.mean(
        torch.abs(predictions - true_values)
    )

    correlation = torch.corrcoef(
        torch.stack([
            predictions.flatten(),
            true_values.flatten(),
        ])
    )[0, 1]

    return pd.Series({
        "MSE": mse.item(),
        "RMSE": rmse.item(),
        "MAE": mae.item(),
        "Correlation": correlation.item(),
    })


def compute_species_error_metrics(
    predictions,
    true_values,
):

    species_names = predictions.columns

    predictions = to_tensor(predictions)
    true_values = to_tensor(true_values)

    mse = torch.mean(
        (predictions - true_values) ** 2,
        dim=0,
    )

    rmse = torch.sqrt(mse)

    mae = torch.mean(
        torch.abs(predictions - true_values),
        dim=0,
    )

    correlations = []

    for j in range(predictions.shape[1]):

        corr = torch.corrcoef(
            torch.stack([
                predictions[:, j],
                true_values[:, j],
            ])
        )[0, 1]

        correlations.append(corr)

    correlations = torch.stack(correlations)

    return pd.DataFrame({
        "MSE": mse.cpu().numpy(),
        "RMSE": rmse.cpu().numpy(),
        "MAE": mae.cpu().numpy(),
        "Correlation": correlations.cpu().numpy(),
    },
    index=species_names)


def plot_metric_boxplots(
    metric_dfs,
    labels,
    metrics=None,
    figsize=None,
):

    if metrics is None:
        metrics = metric_dfs[0].columns.tolist()

    n_metrics = len(metrics)

    if figsize is None:
        figsize = (4 * n_metrics, 4)

    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=figsize,
    )

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):

        data = [
            df[metric].dropna()
            for df in metric_dfs
        ]

        ax.boxplot(
            data,
            labels=labels,
        )

        ax.set_title(metric)

    plt.tight_layout()

    plt.show()