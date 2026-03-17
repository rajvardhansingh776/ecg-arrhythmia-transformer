import os
import wfdb
import numpy as np
import pandas as pd

def load_ptbxl(path):

    meta = pd.read_csv(os.path.join(path,"ptbxl_database.csv"))

    X=[]
    y=[]
    patients=[]

    for _,row in meta.iterrows():

        file=os.path.join(path,row["filename_lr"])

        signal,_=wfdb.rdsamp(file)

        X.append(signal[:,0])

        y.append(row["patient_id"] % 5)

        patients.append(row["patient_id"])

    return np.array(X),np.array(y),np.array(patients)