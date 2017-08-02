import scipy as sp
import numpy as np
import sys

sys.path.append("C:/Users/Yimeng/Documents/GitHub/")
import neurodsp
from neurodsp import shape

def preprocess_filters(x, Fs):

    """ A method that filter the original signal with low pass filter of 200Hz,
    high pass filter of 2Hz and notch filter at 60Hz, 120Hz, 180Hz.

    Parameter:
        x :
        The original signal
        Fs:
        The sample rate

    Return:
        x_notch:
        The filtered signals
    """
    # Low pass at 200Hz
    x_lo = neurodsp.filter(x, Fs, 'lowpass', f_lo=200, N_seconds=.1)

    # Highpass at 2Hz - figure out order
    x_hi = neurodsp.filter(x_lo, Fs, 'highpass', f_hi=2, N_seconds=2)

    # Notch filter at 60Hz, 120Hz and 180Hz
    N_seconds = .5
    x_notch = neurodsp.filter(x_hi, Fs, 'bandstop', f_lo=58, f_hi=62, N_seconds=N_seconds)
    x_notch = neurodsp.filter(x_notch, Fs, 'bandstop', f_lo=118, f_hi=122, N_seconds=N_seconds)
    x_notch = neurodsp.filter(x_notch, Fs, 'bandstop', f_lo=178, f_hi=182, N_seconds=N_seconds)

    return x_notch

def amp_by_trial(x, F_range, Fs, samp_lims, trial_starts):
    x_filt,_=nonshape.bandpass_default(x,F_range,Fs,rmv_edge=False)
    beta_amp = np.abs(sp.signal.hilbert(x_filt))

    N_trials = len(trial_starts)
    N_samps = samp_lims[1] - samp_lims[0]

    beta_amps = np.zeros((N_trials,N_samps))
    for i, t in enumerate(trial_starts):
        beta_amps[i] = beta_amp[t+samp_lims[0]:t+samp_lims[1]]

    return beta_amps
