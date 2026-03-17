from scipy.signal import butter,filtfilt

def bandpass(signal,fs=500):

    b,a=butter(4,[0.5/(0.5*fs),40/(0.5*fs)],btype='band')

    return filtfilt(b,a,signal)