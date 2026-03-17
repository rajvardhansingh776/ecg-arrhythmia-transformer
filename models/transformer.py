import torch
import torch.nn as nn

class ECGTransformer(nn.Module):

    def __init__(self,seq_len,classes):

        super().__init__()

        self.embed=nn.Linear(1,128)

        encoder=nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            batch_first=True
        )

        self.encoder=nn.TransformerEncoder(
            encoder,3
        )

        self.fc=nn.Linear(128,classes)

    def forward(self,x):

        x=x.unsqueeze(-1)

        x=self.embed(x)

        x=self.encoder(x)

        x=x.mean(dim=1)

        return self.fc(x)