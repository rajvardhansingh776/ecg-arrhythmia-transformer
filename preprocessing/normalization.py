import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize(X):

    scaler=StandardScaler()

    X=scaler.fit_transform(X.reshape(len(X),-1))

    return X.reshape(len(X),-1)