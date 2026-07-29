import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def evaluate_history(
    history: np.ndarray,
    model_name: str,
    save_path: str | None = None,
) -> None:
    epochs = history[:, 0]
    train_accuracy = history[:, 2]
    test_accuracy = history[:, 4]

    plt.figure(figsize=(16, 10))

    plt.plot(
        epochs,
        train_accuracy,
        c="k",
        label="Train accuracy",
    )
    plt.plot(
        epochs,
        test_accuracy,
        c="b",
        label="Test accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Test Accuracy ({model_name})")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
    plt.close()


def plot_loss(
    history: np.ndarray,
    model_name: str,
    save_path: str | None = None,
) -> None:
    epochs = history[:, 0]
    train_loss = history[:, 1]
    test_loss = history[:, 3]

    plt.figure(figsize=(16, 10))

    plt.plot(
        epochs,
        train_loss,
        label="Train loss",
    )
    plt.plot(
        epochs,
        test_loss,
        label="Test loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Test Loss ({model_name})")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
    plt.close()