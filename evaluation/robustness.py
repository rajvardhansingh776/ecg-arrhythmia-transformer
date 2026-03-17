import numpy as np
from sklearn.metrics import f1_score

def add_noise(x,level):

    noise=np.random.normal(0,level,x.shape)

    return x+noise


def robustness_test(model,X_test,y_test,device,levels=[0.01,0.03,0.05]):

    results={}

    for l in levels:

        noisy=add_noise(X_test,l)

        preds=[]

        for x in noisy:

            import torch

            x=torch.tensor(x).unsqueeze(0).float().to(device)

            p=model(x)

            preds.append(p.argmax(1).item())

        f1=f1_score(y_test,preds,average="macro")

        results[l]=f1

    return results