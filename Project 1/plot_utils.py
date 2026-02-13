import os
import matplotlib.pyplot as plt


def _finalize(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_losses(epochs, ce_losses, total_losses, title="Training Losses", save_path=None):
    fig, ax = plt.subplots()
    ax.plot(epochs, ce_losses, label="Cross-Entropy Loss", linewidth=1.5)
    ax.plot(epochs, total_losses, label="Total Loss (CE + L2)", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)
    _finalize(fig, save_path)


def plot_error_rates(epochs, train_errors, test_errors, title="Error Rates", save_path=None):
    fig, ax = plt.subplots()
    ax.plot(epochs, train_errors, label="Train Error", linewidth=1.5)
    ax.plot(epochs, test_errors, label="Test Error", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Misclassification Rate")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)
    _finalize(fig, save_path)


def plot_test_error(runs, title="Testing Error Rate", save_path=None):
    """Plot multiple test-error curves for comparing hyperparameter configs.

    Parameters
    ----------
    runs : list[dict]
        Each dict must contain:
            "epochs"      – list of epoch numbers
            "error_rates" – list of test error values
            "label"       – string describing the config (e.g. "lr=0.01, m=0.8")
    title : str
    save_path : str or None
    """
    fig, ax = plt.subplots()
    for run in runs:
        ax.plot(run["epochs"], run["error_rates"], label=run["label"], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Misclassification Rate")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)
    _finalize(fig, save_path)


def plot_all(history, title_prefix="", save_dir=None):
    """Convenience wrapper that plots losses and error rates.

    Parameters
    ----------
    history : dict
        Expected keys: "ce_loss", "total_loss", "train_error", "test_error".
        Each value is a list of per-epoch metric values.
    title_prefix : str
        Prepended to each plot title (e.g. "Run 1 – ").
    save_dir : str or None
        Directory to save PNGs into. If None, plots are shown interactively.
    """
    epochs = list(range(1, len(history["ce_loss"]) + 1))

    loss_path = os.path.join(save_dir, "losses.png") if save_dir else None
    plot_losses(epochs, history["ce_loss"], history["total_loss"],
                title=f"{title_prefix}Training Losses", save_path=loss_path)

    err_path = os.path.join(save_dir, "error_rates.png") if save_dir else None
    plot_error_rates(epochs, history["train_error"], history["test_error"],
                     title=f"{title_prefix}Error Rates", save_path=err_path)