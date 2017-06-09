import scipy as sp
import numpy as np
import sys
#sys.path.append("C:/Users/user/Desktop/ipynb/misshapen/")
sys.path.append("C:/Users/Yimeng/Documents/GitHub/")
import misshapen
from misshapen import nonshape

def preprocess_filters(x,Fs):
    x_nonoise = nonshape.notch_default(x, 60, 2, Fs)
    x_nonoise1=nonshape.notch_default(x_nonoise,120,2,Fs)
    x_nonoise2=nonshape.notch_default(x_nonoise1,180,2,Fs)

    x_nonoise3=nonshape.lowpass_default(x_nonoise2,Fs,200,500)
    x_nonoise4=nonshape.highpass_default(x_nonoise3,Fs,0.5,2001)

    return x_nonoise4

def amp_by_trial(x, F_range, Fs, samp_lims, trial_starts):
    x_filt,_=nonshape.bandpass_default(x,F_range,Fs,rmv_edge=False)
    beta_amp = np.abs(sp.signal.hilbert(x_filt))

    N_trials = len(trial_starts)
    N_samps = samp_lims[1] - samp_lims[0]

    beta_amps = np.zeros((N_trials,N_samps))
    for i, t in enumerate(trial_starts):
        beta_amps[i] = beta_amp[t+samp_lims[0]:t+samp_lims[1]]

    return beta_amps
