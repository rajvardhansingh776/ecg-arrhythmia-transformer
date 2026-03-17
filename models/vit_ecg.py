import torch
import torch.nn as nn

class ECGPatchEmbedding(nn.Module):

    def __init__(self,seq_len,patch=50,dim=128):

        super().__init__()

        self.patch=patch
        self.n=seq_len//patch

        self.linear=nn.Linear(patch,dim)

    def forward(self,x):

        B,L=x.shape

        x=x[:,:self.n*self.patch]

        x=x.view(B,self.n,self.patch)

        return self.linear(x)

class ViTECG(nn.Module):

    def __init__(self,seq_len,classes,patch=50):

        super().__init__()

        self.patch=ECGPatchEmbedding(seq_len,patch)

        tokens=seq_len//patch

        self.pos=nn.Parameter(torch.randn(1,tokens,128))

        encoder=nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            batch_first=True
        )

        self.transformer=nn.TransformerEncoder(encoder,4)

        self.fc=nn.Linear(128,classes)

    def forward(self,x):

        x=self.patch(x)

        x=x+self.pos

        x=self.transformer(x)

        x=x.mean(dim=1)

        return self.fc(x)