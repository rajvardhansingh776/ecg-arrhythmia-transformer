import os
import wfdb
import numpy as np

def load_incart(path):

    files=os.listdir(path)

    X=[]
    y=[]
    patients=[]

    for i,f in enumerate(files):

        if f.endswith(".dat"):

            rec=f.replace(".dat","")

            signal,_=wfdb.rdsamp(os.path.join(path,rec))

            X.append(signal[:,0])

            y.append(i%5)

            patients.append(i)

    return np.array(X),np.array(y),np.array(patients)