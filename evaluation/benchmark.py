import torch
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,balanced_accuracy_score

def benchmark(model,loader,device):

    model.eval()

    preds=[]
    labels=[]
    probs=[]

    with torch.no_grad():

        for X,y in loader:

            X=X.to(device)

            out=model(X)

            p=torch.softmax(out,1)

            preds.extend(torch.argmax(p,1).cpu().numpy())
            probs.extend(p.cpu().numpy())
            labels.extend(y.numpy())

    preds=np.array(preds)
    labels=np.array(labels)
    probs=np.array(probs)

    acc=accuracy_score(labels,preds)
    f1=f1_score(labels,preds,average="macro")
    bal=balanced_accuracy_score(labels,preds)

    try:
        auc=roc_auc_score(labels,probs,multi_class="ovr")
    except:
        auc=0

    metrics={
        "accuracy":acc,
        "f1":f1,
        "balanced_accuracy":bal,
        "auroc":auc
    }

    return metrics,preds,labels,probs