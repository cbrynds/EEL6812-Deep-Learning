import matplotlib.pyplot as plt
import os

def validate_inputs(baseline_values, modified_values, num_epochs):
    if num_epochs <= 0:
        raise ValueError("num_epochs must be a positive integer.")
    if len(modified_values) != num_epochs:
        raise ValueError(
            f"modified_values length ({len(modified_values)}) must match num_epochs ({num_epochs})."
        )
    if baseline_values and len(baseline_values) != num_epochs:
        raise ValueError(
            f"baseline_values length ({len(baseline_values)}) must match num_epochs ({num_epochs}) when provided."
        )


def plot_metric(baseline_values, modified_values, num_epochs, title, output_dir):
    validate_inputs(baseline_values, modified_values, num_epochs)

    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(8, 5))

    if baseline_values:
        plt.plot(epochs, baseline_values, marker="o", label="Baseline")
    plt.plot(epochs, modified_values, marker="o", label="Modified")

    plt.xlabel("Epoch")
    plt.ylabel("Loss / Error Rate")
    # plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(f"{output_dir}/{title}.png")


def plot_total_loss(baseline_loss, modified_loss, num_epochs, output_dir):
    plot_metric(baseline_loss, modified_loss, num_epochs, "Total Loss vs Epoch", output_dir)


def plot_training_accuracy(baseline_train_accuracy, modified_train_accuracy, num_epochs, output_dir):
    plot_metric(
        baseline_train_accuracy,
        modified_train_accuracy,
        num_epochs,
        "Training Accuracy vs Epoch",
        output_dir
    )


def plot_testing_accuracy(baseline_test_accuracy, modified_test_accuracy, num_epochs, output_dir):
    plot_metric(
        baseline_test_accuracy,
        modified_test_accuracy,
        num_epochs,
        "Testing Accuracy vs Epoch",
        output_dir
    )
