import torch
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
from sklearn.utils.class_weight import compute_class_weight

from models.transformer import ECGTransformer
from models.patch_transformer import PatchECGTransformer
from models.vit_ecg import ViTECG
from models.multilead_transformer import MultiLeadTransformer

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

X=np.load("data/processed/ptbxl_signals.npy")
y=np.load("data/processed/ptbxl_labels.npy")

dataset=TensorDataset(torch.tensor(X).float(),torch.tensor(y))

loader=DataLoader(dataset,64,shuffle=True)

MODEL_TYPE="vit"

if MODEL_TYPE=="transformer":
    model=ECGTransformer(seq_len=X.shape[1],classes=5)
elif MODEL_TYPE=="patch":
    model=PatchECGTransformer(seq_len=X.shape[1],classes=5)
elif MODEL_TYPE=="vit":
    model=ViTECG(seq_len=X.shape[1],classes=5)
elif MODEL_TYPE=="multilead":
    model=MultiLeadTransformer(leads=12,classes=5)

model=model.to(device)

weights=compute_class_weight("balanced",classes=np.unique(y),y=y)
weights=torch.tensor(weights).float().to(device)

loss_fn=torch.nn.CrossEntropyLoss(weight=weights)

opt=torch.optim.Adam(model.parameters(),1e-4)

for epoch in range(20):

    model.train()

    for xb,yb in loader:

        xb,yb=xb.to(device),yb.to(device)

        opt.zero_grad()

        out=model(xb)

        loss=loss_fn(out,yb)

        loss.backward()

        opt.step()

    print("epoch",epoch)