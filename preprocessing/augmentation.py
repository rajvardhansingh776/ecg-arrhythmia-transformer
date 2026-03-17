import numpy as np

def jitter(x,sigma=0.03):

    noise=np.random.normal(0,sigma,x.shape)

    return x+noise


def scaling(x,sigma=0.1):

    factor=np.random.normal(1.0,sigma)

    return x*factor


def random_crop(x,crop=0.9):

    length=int(len(x)*crop)

    start=np.random.randint(0,len(x)-length)

    return x[start:start+length]