import matplotlib.pyplot as plt
import os

PLOT_TITLE_FONTSIZE = 18
PLOT_LABEL_FONTSIZE = 15
PLOT_TICK_FONTSIZE = 13
PLOT_LEGEND_FONTSIZE = 11
PLOT_LEGEND_TITLE_FONTSIZE = 12


def image_label_from_path(img_path):
    # Use image id (filename without extension) instead of full path in legend.
    return os.path.splitext(os.path.basename(img_path))[0]

def linestyle_for_index(idx):
    linestyles = ["-", "--", "-.", ":"]
    return linestyles[idx % len(linestyles)]


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


def plot_metric(baseline_values, modified_values, num_epochs, title, y_metric, output_dir):
    validate_inputs(baseline_values, modified_values, num_epochs)

    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(8, 5))

    if baseline_values:
        plt.plot(epochs, baseline_values, marker="o", label="Baseline")
    plt.plot(epochs, modified_values, marker="o", label="Modified")

    plt.xlabel("Epoch")
    plt.ylabel(y_metric)
    # plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(f"{output_dir}/{title}.png")


def plot_total_loss(baseline_loss, modified_loss, num_epochs, output_dir):
    plot_metric(baseline_loss, modified_loss, num_epochs, "Total Loss vs Epoch", "Loss", output_dir)


def plot_training_error_rate(baseline_train_error_rate, modified_train_error_rate, num_epochs, output_dir):
    plot_metric(
        baseline_train_error_rate,
        modified_train_error_rate,
        num_epochs,
        "Training Error Rate vs Epoch",
        "Error Rate",
        output_dir
    )


def plot_testing_error_rate(baseline_test_error_rate, modified_test_error_rate, num_epochs, output_dir):
    plot_metric(
        baseline_test_error_rate,
        modified_test_error_rate,
        num_epochs,
        "Testing Error Rate vs Epoch",
        "Error Rate",
        output_dir
    )
    
def plot_detections_vs_conf(num_detections_conf, fixed_iou, save_dir):
    conf_values = sorted(num_detections_conf.keys())
    if len(conf_values) == 0:
        return

    image_paths = sorted(num_detections_conf[conf_values[0]].keys())
    cmap = plt.get_cmap("tab20", max(1, len(image_paths)))

    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, img_path in enumerate(image_paths):
        y = [num_detections_conf[conf].get(img_path, 0) for conf in conf_values]
        ax.plot(
            conf_values,
            y,
            label=image_label_from_path(img_path),
            color=cmap(idx),
            linestyle=linestyle_for_index(idx),
            linewidth=2,
        )

    ax.set_xlabel("Confidence Threshold", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel("Number of Detections", fontsize=PLOT_LABEL_FONTSIZE)
    # ax.set_title(f"Detections vs Confidence (fixed IoU={fixed_iou})", fontsize=PLOT_TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    # ax.legend(
    #     title="Image ID",
    #     ncol=2,
    #     fontsize=PLOT_LEGEND_FONTSIZE,
    #     title_fontsize=PLOT_LEGEND_TITLE_FONTSIZE,
    #     loc="center left",
    #     bbox_to_anchor=(1.02, 0.5),
    #     borderaxespad=0.0,
    # )
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig.savefig(
        os.path.join(save_dir, f"detections_vs_conf_iou_{fixed_iou}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

def plot_detections_vs_iou(num_detections_iou, fixed_conf, save_dir):
    iou_values = sorted(num_detections_iou.keys())
    if len(iou_values) == 0:
        return

    image_paths = sorted(num_detections_iou[iou_values[0]].keys())
    cmap = plt.get_cmap("tab20", max(1, len(image_paths)))

    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, img_path in enumerate(image_paths):
        y = [num_detections_iou[iou].get(img_path, 0) for iou in iou_values]
        ax.plot(
            iou_values,
            y,
            label=image_label_from_path(img_path),
            color=cmap(idx),
            linestyle=linestyle_for_index(idx),
            linewidth=2,
        )

    ax.set_xlabel("IoU Threshold", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel("Number of Detections", fontsize=PLOT_LABEL_FONTSIZE)
    # ax.set_title(f"Detections vs IoU (fixed confidence={fixed_conf})", fontsize=PLOT_TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)
    # ax.legend(
    #     # title="Image ID",
    #     ncol=2,
    #     fontsize=PLOT_LEGEND_FONTSIZE,
    #     title_fontsize=PLOT_LEGEND_TITLE_FONTSIZE,
    #     loc="center left",
    #     bbox_to_anchor=(1.02, 0.5),
    #     borderaxespad=0.0,
    # )
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig.savefig(
        os.path.join(save_dir, f"detections_vs_iou_conf_{fixed_conf}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
