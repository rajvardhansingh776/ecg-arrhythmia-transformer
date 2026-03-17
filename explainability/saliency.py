import torch
import matplotlib.pyplot as plt

def saliency(model,x):

    x=torch.tensor(x).unsqueeze(0).float()

    x.requires_grad=True

    y=model(x)

    y.max().backward()

    grad=x.grad.abs().detach().numpy()[0]

    plt.plot(grad)

    plt.savefig("results/figures/saliency.png")