"""
Cory Brynds

Included plotting utilities for visualizing results, by default outputted to /results
"""

import os
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
})

def save_fig_to_file(fig, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# Plot curves for cross-entropy loss and total loss over epochs
def plot_losses(epochs, ce_losses, total_losses, title="Training Losses", save_path=None):
    fig, ax = plt.subplots()
    ax.plot(epochs, ce_losses, label="Cross-Entropy Loss", linewidth=1.5)
    ax.plot(epochs, total_losses, label="Total Loss (CE + L2)", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)
    save_fig_to_file(fig, save_path)

# Plot both training and test error rates over epochs
def plot_error_rates(epochs, train_errors, test_errors, title="Error Rates", save_path=None):
    fig, ax = plt.subplots()
    ax.plot(epochs, train_errors, label="Train Error", linewidth=1.5)
    ax.plot(epochs, test_errors, label="Test Error", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Misclassification Rate")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)
    save_fig_to_file(fig, save_path)


def plot_sweep_errors(runs, title="Error Rate Comparison", save_path=None):
    fig, (ax_train, ax_test) = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

    for run in runs:
        ax_train.plot(run["epochs"], run["train_error_rates"], label=run["label"], linewidth=1.5)
        ax_test.plot(run["epochs"], run["error_rates"], label=run["label"], linewidth=1.5)

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Misclassification Rate")
    ax_train.set_title("Training Error")
    ax_train.legend(loc="best")
    ax_train.grid(True, linestyle="--", alpha=0.6)

    ax_test.set_xlabel("Epoch")
    ax_test.set_title("Testing Error")
    ax_test.legend(loc="best")
    ax_test.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    save_fig_to_file(fig, save_path)


def plot_frobenius_norms(runs, title="Frobenius Norm of Weights", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for run in runs:
        ax1.plot(run["epochs"], run["w1_frobenius"], label=run["label"], linewidth=1.5)
        ax2.plot(run["epochs"], run["w2_frobenius"], label=run["label"], linewidth=1.5)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r"$\|W^{(1)}\|_F$")
    ax1.set_title(r"$W^{(1)}$ Frobenius Norm")
    ax1.legend(loc="best")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"$\|w^{(2)}\|_2$")
    ax2.set_title(r"$w^{(2)}$ L2 Norm")
    ax2.legend(loc="best")
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(title)
    fig.tight_layout()
    save_fig_to_file(fig, save_path)


def plot_all(history, title_prefix="", save_dir=None):
    epochs = list(range(1, len(history["ce_loss"]) + 1))

    loss_path = os.path.join(save_dir, "losses.png") if save_dir else None
    plot_losses(epochs, history["ce_loss"], history["total_loss"],
                title=f"{title_prefix}Training Losses", save_path=loss_path)

    err_path = os.path.join(save_dir, "error_rates.png") if save_dir else None
    plot_error_rates(epochs, history["train_error"], history["test_error"],
                     title=f"{title_prefix}Error Rates", save_path=err_path)