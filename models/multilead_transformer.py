import torch
import torch.nn as nn


class MultiLeadEmbedding(nn.Module):

    def __init__(self,leads,dim):

        super().__init__()

        self.embed=nn.Linear(1,dim)

        self.leads=leads

    def forward(self,x):

        B,L,T=x.shape

        x=x.view(B*L,T,1)

        x=self.embed(x)

        x=x.mean(dim=1)

        x=x.view(B,L,-1)

        return x


class MultiLeadTransformer(nn.Module):

    def __init__(self,leads=12,classes=5):

        super().__init__()

        self.embed=MultiLeadEmbedding(leads,128)

        encoder=nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            batch_first=True
        )

        self.transformer=nn.TransformerEncoder(
            encoder,
            num_layers=3
        )

        self.fc=nn.Linear(128,classes)

    def forward(self,x):

        x=self.embed(x)

        x=self.transformer(x)

        x=x.mean(dim=1)

        return self.fc(x)