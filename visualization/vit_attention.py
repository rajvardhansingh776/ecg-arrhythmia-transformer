import torch
import matplotlib.pyplot as plt


def visualize_attention(model,signal):

    signal=torch.tensor(signal).unsqueeze(0).float()

    signal.requires_grad=True

    out=model(signal)

    out.max().backward()

    attention=signal.grad.abs().detach().numpy()[0]

    plt.plot(attention)

    plt.title("Transformer Attention Map")

    plt.savefig("results/figures/vit_attention.png")

    plt.show()