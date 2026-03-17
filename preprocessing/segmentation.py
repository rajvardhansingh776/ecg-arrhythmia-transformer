import numpy as np
from scipy.signal import find_peaks

def segment_beats(signal,window=180):

    peaks,_=find_peaks(signal,distance=200)

    beats=[]

    for p in peaks:

        start=p-window
        end=p+window

        if start>0 and end<len(signal):

            beats.append(signal[start:end])

    return np.array(beats)