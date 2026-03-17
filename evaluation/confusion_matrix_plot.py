import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(labels,preds):

    cm=confusion_matrix(labels,preds)

    plt.figure(figsize=(6,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title("Confusion Matrix")

    plt.savefig("results/figures/confusion_matrix.png")

    plt.show()