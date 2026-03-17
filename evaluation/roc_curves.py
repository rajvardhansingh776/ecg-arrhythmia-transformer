import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize


def plot_roc_curves(labels,probs,n_classes):

    labels=label_binarize(labels,classes=list(range(n_classes)))

    plt.figure(figsize=(8,6))

    for i in range(n_classes):

        fpr,tpr,_=roc_curve(labels[:,i],probs[:,i])

        roc_auc=auc(fpr,tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"Class {i} (AUC={roc_auc:.3f})"
        )

    plt.plot([0,1],[0,1],'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC-AUC Curves per Arrhythmia Class")

    plt.legend()

    plt.savefig("results/figures/roc_curves.png")

    plt.show()