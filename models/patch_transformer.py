import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):

    def __init__(self,seq_len,patch=50,dim=128):

        super().__init__()

        self.patch=patch

        self.linear=nn.Linear(patch,dim)

        self.n=seq_len//patch

    def forward(self,x):

        B,L=x.shape

        x=x[:,:self.n*self.patch]

        x=x.view(B,self.n,self.patch)

        return self.linear(x)


class PatchECGTransformer(nn.Module):

    def __init__(self,seq_len,classes):

        super().__init__()

        self.patch=PatchEmbedding(seq_len)

        self.pos=nn.Parameter(torch.randn(1,seq_len//50,128))

        encoder=nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            batch_first=True
        )

        self.transformer=nn.TransformerEncoder(
            encoder,4
        )

        self.fc=nn.Linear(128,classes)

    def forward(self,x):

        x=self.patch(x)

        x=x+self.pos

        x=self.transformer(x)

        x=x.mean(dim=1)

        return self.fc(x)