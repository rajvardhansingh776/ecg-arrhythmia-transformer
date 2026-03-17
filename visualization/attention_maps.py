import torch
import matplotlib.pyplot as plt

def attention_map(model,signal):

    signal=torch.tensor(signal).unsqueeze(0).float()

    signal.requires_grad=True

    out=model(signal)

    out.max().backward()

    att=signal.grad.abs().detach().numpy()[0]

    plt.figure(figsize=(10,4))

    plt.plot(signal.detach().numpy()[0],label="ECG")

    plt.plot(att,label="Attention")

    plt.legend()

    plt.title("Attention Map")

    plt.savefig("results/figures/attention_map.png")