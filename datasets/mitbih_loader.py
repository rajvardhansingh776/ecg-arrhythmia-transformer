import wfdb
import numpy as np

def load_mitbih(records_path,records):

    X=[]
    y=[]
    patients=[]

    for r in records:

        signal,_=wfdb.rdsamp(records_path+"/"+r)

        X.append(signal[:,0])

        y.append(int(r)%5)

        patients.append(r)

    return np.array(X),np.array(y),np.array(patients)