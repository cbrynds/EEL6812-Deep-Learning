import os

import matplotlib.pyplot as plt
import torch

RESULTS_DIR = "results"
TRANSFORMER_RESULTS_DIR = "transformer_results"
VAE_RESULTS_DIR = "vae_results"


def series_from_plot_data(plot_data, split, metric_index):
    points = plot_data.get(split, [])
    if not points:
        return [], []

    x_values = [point[0] for point in points]
    y_values = [point[metric_index] for point in points]
    return x_values, y_values


def plot_metric(model_plot_data, split, metric_index, title, y_label, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))

    model_labels = ("LSTM", "RNN", "GRU")
    model_colors = ("tab:blue", "tab:orange", "tab:green")

    for label, color, plot_data in zip(model_labels, model_colors, model_plot_data):
        x_values, y_values = series_from_plot_data(plot_data, split, metric_index)
        if x_values:
            plt.plot(x_values, y_values, label=label, color=color, linewidth=2)

    # plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{filename}", dpi=300, bbox_inches="tight")
    plt.close()


def plot_train_loss(lstm_plot_data, rnn_plot_data, gru_plot_data):
    plot_metric(
        (lstm_plot_data, rnn_plot_data, gru_plot_data),
        split="train",
        metric_index=1,
        title="Training Loss",
        y_label="Loss",
        filename="train_loss.png",
    )

def plot_test_loss(lstm_plot_data, rnn_plot_data, gru_plot_data):
    plot_metric(
        (lstm_plot_data, rnn_plot_data, gru_plot_data),
        split="test",
        metric_index=1,
        title="Test Loss",
        y_label="Loss",
        filename="test_loss.png",
    )


def plot_train_accuracy(lstm_plot_data, rnn_plot_data, gru_plot_data):
    plot_metric(
        (lstm_plot_data, rnn_plot_data, gru_plot_data),
        split="train",
        metric_index=2,
        title="Training Accuracy",
        y_label="Accuracy",
        filename="train_accuracy.png",
    )


def plot_test_accuracy(lstm_plot_data, rnn_plot_data, gru_plot_data):
    plot_metric(
        (lstm_plot_data, rnn_plot_data, gru_plot_data),
        split="test",
        metric_index=2,
        title="Test Accuracy",
        y_label="Accuracy",
        filename="test_accuracy.png",
    )

def plot_validation_loss_curve(val_history, run_label, num_heads, num_layers, context_length, with_pos_enc=True, with_causal_mask=True):
    os.makedirs(TRANSFORMER_RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    epochs = [point[0] for point in val_history]
    losses = [point[1] for point in val_history]

    plt.plot(epochs, losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    # plt.title(f"Validation Loss Curve - {run_label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_title = f"val_loss_{run_label}_heads_{num_heads}_layers_{num_layers}_context_{context_length}"
    if not with_pos_enc:
        fig_title += "_no_pos_enc"
    if not with_causal_mask:
        fig_title += "_no_causal_mask"

    plt.savefig(f"{TRANSFORMER_RESULTS_DIR}/{fig_title}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_loss_curve(train_history, run_label, num_heads, num_layers, context_length, with_pos_enc=True, with_causal_mask=True):
    os.makedirs(TRANSFORMER_RESULTS_DIR, exist_ok=True)
    epochs = [point[0] for point in train_history]
    losses = [point[1] for point in train_history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_title = f"train_loss_{run_label}_heads_{num_heads}_layers_{num_layers}_context_{context_length}"
    if not with_pos_enc:
        fig_title += "_no_pos_enc"
    if not with_causal_mask:
        fig_title += "_no_causal_mask"

    plt.savefig(f"{TRANSFORMER_RESULTS_DIR}/{fig_title}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss_comparison(run_histories, metric_key, y_label, filename):
    os.makedirs(TRANSFORMER_RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(9, 5.5))

    for run_label, history in run_histories:
        if not history:
            continue
        epochs = [point[0] for point in history]
        losses = [point[1] for point in history]
        plt.plot(epochs, losses, linewidth=2, label=run_label)

    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{TRANSFORMER_RESULTS_DIR}/{filename}", dpi=300, bbox_inches="tight")
    plt.close()


def plot_vae_loss(plot_data, loss_label = "BCE Reconstruction Loss", filename="vae_loss.png"):
    os.makedirs(VAE_RESULTS_DIR, exist_ok=True)
    epochs = list(range(1, len(plot_data) + 1))
    total_loss = [point["loss"] for point in plot_data]
    reconstruction_loss = [point["reconstruction_loss"] for point in plot_data]
    kld_loss = [point["kld_loss"] for point in plot_data]

    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, total_loss, linewidth=2, label="Total Loss", color="tab:blue")
    plt.plot(epochs, reconstruction_loss, linewidth=2, label=loss_label, color="tab:orange")
    plt.plot(epochs, kld_loss, linewidth=2, label="KL Divergence Loss", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{VAE_RESULTS_DIR}/{filename}", dpi=300, bbox_inches="tight")
    plt.close()


def plot_vae_latent_dim_comparison(run_histories, filename="latent_dim_loss_comparison.png"):
    os.makedirs(VAE_RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(9, 5.5))

    for latent_dim, plot_data in run_histories:
        epochs = list(range(1, len(plot_data) + 1))
        total_loss = [point["loss"] for point in plot_data]
        plt.plot(epochs, total_loss, linewidth=2, label=f"Latent Dim = {latent_dim}")

    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{VAE_RESULTS_DIR}/{filename}", dpi=300, bbox_inches="tight")
    plt.close()


def plot_attention(attn_maps, tokens, layer_idx=0, head_idx=0, num_heads=4, num_layers=2, context_length=64, with_pos_enc=True, with_causal_mask=True, output_subdir="attention_maps"):
    """
    attn_maps[layer_idx]: [1, heads, T, T]
    """
    attn = attn_maps[layer_idx][0, head_idx]  # [T, T]

    plt.figure(figsize=(8, 6))
    plt.imshow(attn, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel("Key positions (attended to)")
    plt.ylabel("Query positions")
    plt.title(f"Attention Map - Layer {layer_idx}, Head {head_idx}")
    plt.tight_layout()
    
    fig_title = f"attention_layer_{layer_idx}_head_{head_idx}_heads_{num_heads}_layers_{num_layers}_context_{context_length}"
    if not with_pos_enc:
        fig_title += "_no_pos_enc"
    if not with_causal_mask:
        fig_title += "_no_causal_mask"
        
    os.makedirs(f"{TRANSFORMER_RESULTS_DIR}/{output_subdir}", exist_ok=True)
    plt.savefig(f"{TRANSFORMER_RESULTS_DIR}/{output_subdir}/{fig_title}.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_avg_attention(attn_maps, tokens, layer_idx=0, num_heads=4, num_layers=2, context_length=64, with_pos_enc=True, with_causal_mask=True, output_subdir="attention_maps"):
    # [1, heads, T, T] -> average over heads -> [T, T]
    attn = attn_maps[layer_idx][0].mean(dim=0)

    plt.figure(figsize=(8, 6))
    plt.imshow(attn, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel("Key positions (attended to)")
    plt.ylabel("Query positions")
    plt.title(f"Average Attention - Layer {layer_idx}")
    plt.tight_layout()

    fig_title = f"average_attention_layer_{layer_idx}_heads_{num_heads}_layers_{num_layers}_context_{context_length}"
    if not with_pos_enc:
        fig_title += "_no_pos_enc"
    if not with_causal_mask:
        fig_title += "_no_causal_mask"
        
    os.makedirs(f"{TRANSFORMER_RESULTS_DIR}/{output_subdir}", exist_ok=True)
    plt.savefig(f"{TRANSFORMER_RESULTS_DIR}/{output_subdir}/{fig_title}.png", dpi=300, bbox_inches="tight")
    plt.close()

# Based on the equation in the assignment description
def compute_attention_entropy(attn_maps):
    entropies = []

    for layer_attn in attn_maps:
        probs = layer_attn.clamp_min(1e-12)
        layer_entropy = -(probs * torch.log(probs)).sum(dim=-1)
        entropies.append(layer_entropy.mean().item())

    return entropies
