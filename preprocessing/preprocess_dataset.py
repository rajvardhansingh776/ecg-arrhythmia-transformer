import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from datasets.ptbxl_loader import load_ptbxl
from datasets.mitbih_loader import load_mitbih
from datasets.incart_loader import load_incart

from preprocessing.filtering import bandpass
from preprocessing.normalization import normalize
from preprocessing.segmentation import segment_beats

DATA_DIR="data/raw"
PROCESSED_DIR="data/processed"
SPLIT_DIR="data/splits"

os.makedirs(PROCESSED_DIR,exist_ok=True)
os.makedirs(SPLIT_DIR,exist_ok=True)

def preprocess_signals(X):

    processed=[]

    for x in X:

        x=bandpass(x)

        beats=segment_beats(x)

        processed.extend(beats)

    processed=np.array(processed)

    processed=normalize(processed)

    return processed

def save_processed(name,X,y):

    np.save(os.path.join(PROCESSED_DIR,f"{name}_signals.npy"),X)
    np.save(os.path.join(PROCESSED_DIR,f"{name}_labels.npy"),y)

def create_split(X,y,patients,name):

    splitter=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    train_idx,test_idx=next(splitter.split(X,y,patients))

    np.save(os.path.join(SPLIT_DIR,f"{name}_train.npy"),train_idx)
    np.save(os.path.join(SPLIT_DIR,f"{name}_test.npy"),test_idx)

def process_ptbxl():

    X,y,patients=load_ptbxl(os.path.join(DATA_DIR,"ptbxl"))

    X=preprocess_signals(X)

    save_processed("ptbxl",X,y)

    create_split(X,y,patients,"ptbxl")

def process_mitbih():

    records=[
"100","101","102","103","104","105","106","107","108","109",
"111","112","113","114","115","116","117","118","119","121",
"122","123","124","200","201","202","203","205","207","208",
"209","210","212","213","214","215","217","219","220","221",
"222","223","228","230","231","232","233","234"
]

    X,y,patients=load_mitbih(os.path.join(DATA_DIR,"mitbih"),records)

    X=preprocess_signals(X)

    save_processed("mitbih",X,y)

    create_split(X,y,patients,"mitbih")

def process_incart():

    X,y,patients=load_incart(os.path.join(DATA_DIR,"incart"))

    X=preprocess_signals(X)

    save_processed("incart",X,y)

    create_split(X,y,patients,"incart")

if __name__=="__main__":

    process_ptbxl()
    process_mitbih()
    process_incart()

    print("Preprocessing complete")