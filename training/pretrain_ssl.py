import torch
from torch.utils.data import DataLoader,TensorDataset
import numpy as np

from models.ssl_encoder import ECGEncoder
from training.contrastive_loss import contrastive_loss
from preprocessing.augmentation import jitter,scaling

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

X=np.load("data/processed/ptbxl_signals.npy")

dataset=TensorDataset(torch.tensor(X).float())

loader=DataLoader(dataset,64,shuffle=True)

model=ECGEncoder().to(device)

opt=torch.optim.Adam(model.parameters(),1e-3)

for epoch in range(20):

    total=0

    for batch in loader:

        x=batch[0]

        x1=jitter(x.numpy())
        x2=scaling(x.numpy())

        x1=torch.tensor(x1).float().to(device)
        x2=torch.tensor(x2).float().to(device)

        z1=model(x1)
        z2=model(x2)

        loss=contrastive_loss(z1,z2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total+=loss.item()

    print("SSL epoch",epoch,total/len(loader))

torch.save(model.state_dict(),"models/ssl_encoder.pt")