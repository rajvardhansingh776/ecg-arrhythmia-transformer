import torch
import torch.nn.functional as F

def contrastive_loss(z1,z2,temp=0.5):

    z1=F.normalize(z1,dim=1)
    z2=F.normalize(z2,dim=1)

    sim=torch.mm(z1,z2.T)/temp

    labels=torch.arange(z1.size(0)).to(z1.device)

    loss=F.cross_entropy(sim,labels)

    return loss