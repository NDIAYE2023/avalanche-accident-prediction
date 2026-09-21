import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def calculate_auc(y_true, y_probability):
    """
    Calculate the Area Under the ROC Curve (AUC).
    """
    return roc_auc_score(
        y_true,
        y_probability
    )


def plot_roc_curve(y_true, y_probability):
    """
    Plot the ROC curve and display the corresponding AUC.
    """

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true,
        y_probability
    )

    auc_score = calculate_auc(
        y_true,
        y_probability
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC curve (AUC = {auc_score:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Avalanche Accident Prediction")
    plt.legend()
    plt.tight_layout()

    plt.show()


def classification_metrics(
    y_true,
    y_probability,
    threshold=0.5
):
    """
    Compute classification metrics using a probability threshold.
    """

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    matrix = confusion_matrix( y_true, y_pred  )

    report = classification_report( y_true,  y_pred )
    return matrix, report


if __name__ == "__main__":
    print(
        "Evaluation module for avalanche accident prediction."
    )
