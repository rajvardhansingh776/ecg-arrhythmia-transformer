import torch
import numpy as np
from torch.utils.data import DataLoader,TensorDataset

from models.vit_ecg import ViTECG
from evaluation.benchmark import benchmark
from evaluation.roc_curves import plot_roc_curves
from evaluation.pr_curves import plot_pr_curves

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

X=np.load("data/processed/ptbxl_signals.npy")
y=np.load("data/processed/ptbxl_labels.npy")

dataset=TensorDataset(torch.tensor(X).float(),torch.tensor(y))
loader=DataLoader(dataset,64)

model=ViTECG(seq_len=X.shape[1],classes=5).to(device)

preds,labels,probs=benchmark(model,loader,device)

plot_roc_curves(labels,probs,5)
plot_pr_curves(labels,probs,5)