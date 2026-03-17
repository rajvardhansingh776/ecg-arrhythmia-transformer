import torch
import torch.nn as nn

class ECGEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv=nn.Sequential(
            nn.Conv1d(1,32,7,padding=3),
            nn.ReLU(),
            nn.Conv1d(32,64,5,padding=2),
            nn.ReLU(),
            nn.Conv1d(64,128,3,padding=1),
            nn.ReLU()
        )

        self.pool=nn.AdaptiveAvgPool1d(1)

    def forward(self,x):

        x=x.unsqueeze(1)

        x=self.conv(x)

        x=self.pool(x)

        return x.squeeze(-1)