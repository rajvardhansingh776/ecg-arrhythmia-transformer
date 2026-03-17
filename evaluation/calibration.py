import numpy as np

def expected_calibration_error(probs,labels,bins=10):

    bin_edges=np.linspace(0,1,bins+1)

    ece=0

    for i in range(bins):

        mask=(probs[:,1]>bin_edges[i])&(probs[:,1]<=bin_edges[i+1])

        if mask.sum()>0:

            acc=(labels[mask]==1).mean()

            conf=probs[mask,1].mean()

            ece+=abs(acc-conf)*mask.mean()

    return ece