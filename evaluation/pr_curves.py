import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.preprocessing import label_binarize


def plot_pr_curves(labels,probs,n_classes):

    labels=label_binarize(labels,classes=list(range(n_classes)))

    plt.figure(figsize=(8,6))

    for i in range(n_classes):

        precision,recall,_=precision_recall_curve(labels[:,i],probs[:,i])

        ap=average_precision_score(labels[:,i],probs[:,i])

        plt.plot(
            recall,
            precision,
            label=f"Class {i} (AP={ap:.3f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision–Recall Curves per Arrhythmia Class")

    plt.legend()

    plt.savefig("results/figures/pr_curves.png")

    plt.show()