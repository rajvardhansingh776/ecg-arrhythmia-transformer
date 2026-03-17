import shap
import torch
import matplotlib.pyplot as plt

def shap_explain(model,X):

    background=torch.tensor(X[:100]).float()

    explainer=shap.DeepExplainer(model,background)

    samples=torch.tensor(X[:20]).float()

    shap_values=explainer.shap_values(samples)

    shap.summary_plot(shap_values,samples.numpy(),show=False)

    plt.savefig("results/figures/shap_summary.png")