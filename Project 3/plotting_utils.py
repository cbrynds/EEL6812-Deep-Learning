from pathlib import Path

import matplotlib.pyplot as plt


MODEL_LABELS = ("LSTM", "RNN", "GRU")
MODEL_COLORS = ("tab:blue", "tab:orange", "tab:green")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _series_from_plot_data(plot_data, split, metric_index):
    points = plot_data.get(split, [])
    if not points:
        return [], []

    x_values = [point[0] for point in points]
    y_values = [point[metric_index] for point in points]
    return x_values, y_values


def _plot_metric(model_plot_data, split, metric_index, title, y_label, filename):
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))

    for label, color, plot_data in zip(MODEL_LABELS, MODEL_COLORS, model_plot_data):
        x_values, y_values = _series_from_plot_data(plot_data, split, metric_index)
        if x_values:
            plt.plot(x_values, y_values, label=label, color=color, linewidth=2)

    # plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_train_loss(lstm_plot_data, rnn_plot_data, gru_plot_data):
    _plot_metric(
        (lstm_plot_data, rnn_plot_data, gru_plot_data),
        split="train",
        metric_index=1,
        title="Training Loss",
        y_label="Loss",
        filename="train_loss.png",
    )


def plot_test_loss(lstm_plot_data, rnn_plot_data, gru_plot_data):
    _plot_metric(
        (lstm_plot_data, rnn_plot_data, gru_plot_data),
        split="test",
        metric_index=1,
        title="Test Loss",
        y_label="Loss",
        filename="test_loss.png",
    )


def plot_train_accuracy(lstm_plot_data, rnn_plot_data, gru_plot_data):
    _plot_metric(
        (lstm_plot_data, rnn_plot_data, gru_plot_data),
        split="train",
        metric_index=2,
        title="Training Accuracy",
        y_label="Accuracy",
        filename="train_accuracy.png",
    )


def plot_test_accuracy(lstm_plot_data, rnn_plot_data, gru_plot_data):
    _plot_metric(
        (lstm_plot_data, rnn_plot_data, gru_plot_data),
        split="test",
        metric_index=2,
        title="Test Accuracy",
        y_label="Accuracy",
        filename="test_accuracy.png",
    )

def plot_attention(attn_maps, tokens, layer_idx=0, head_idx=0):
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
    plt.savefig(f"attention_layer{layer_idx}_head{head_idx}.png")
    
def plot_avg_attention(attn_maps, tokens, layer_idx=0):
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
    plt.savefig(f"average_attention_layer{layer_idx}.png")