import matplotlib.pyplot as plt


MODEL_LABELS = ("LSTM", "RNN", "GRU")
MODEL_COLORS = ("tab:blue", "tab:orange", "tab:green")


def _series_from_plot_data(plot_data, split, metric_index):
    points = plot_data.get(split, [])
    if not points:
        return [], []

    x_values = [point[0] for point in points]
    y_values = [point[metric_index] for point in points]
    return x_values, y_values


def _plot_metric(model_plot_data, split, metric_index, title, y_label):
    plt.figure(figsize=(10, 6))

    for label, color, plot_data in zip(MODEL_LABELS, MODEL_COLORS, model_plot_data):
        x_values, y_values = _series_from_plot_data(plot_data, split, metric_index)
        if x_values:
            plt.plot(x_values, y_values, label=label, color=color, linewidth=2)

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_train_loss(lstm_plot_data, rnn_plot_data, gru_plot_data):
    _plot_metric(
        (lstm_plot_data, rnn_plot_data, gru_plot_data),
        split="train",
        metric_index=1,
        title="Training Loss",
        y_label="Loss",
    )


def plot_test_loss(lstm_plot_data, rnn_plot_data, gru_plot_data):
    _plot_metric(
        (lstm_plot_data, rnn_plot_data, gru_plot_data),
        split="test",
        metric_index=1,
        title="Test Loss",
        y_label="Loss",
    )


def plot_train_accuracy(lstm_plot_data, rnn_plot_data, gru_plot_data):
    _plot_metric(
        (lstm_plot_data, rnn_plot_data, gru_plot_data),
        split="train",
        metric_index=2,
        title="Training Accuracy",
        y_label="Accuracy",
    )


def plot_test_accuracy(lstm_plot_data, rnn_plot_data, gru_plot_data):
    _plot_metric(
        (lstm_plot_data, rnn_plot_data, gru_plot_data),
        split="test",
        metric_index=2,
        title="Test Accuracy",
        y_label="Accuracy",
    )
