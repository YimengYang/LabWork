import glob
import re
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
def load_paths(data_dir='C:/data2/dg/'):
    all_subjects=_get_subjects(data_dir)

    all_regions={}
    for subj in all_subjects:
        all_regions[subj]=_get_region(data_dir,subj)

    data_paths = {}
    for s in all_subjects:
        data_paths[s] = {}
        for r in all_regions[s]:
            data_paths[s][r] = _get_electrode(data_dir, s, r)
    return data_paths, all_regions

def _get_subjects(data_dir,name_len=2):
    all_subjects=[]
    subjects=glob.glob(data_dir+'/*')
    for sub in subjects:
        if len(sub) == len(data_dir)+name_len:
            all_subjects.append(sub[-name_len:])
    return all_subjects

def _get_region(data_dir,subject):
    all_reg=[]
    for reg in glob.glob(data_dir+'/'+subject+'/*'):
        if(reg[-1:].isdigit()):
            all_reg.append(reg[-1:])
    return all_reg

def _get_electrode(data_dir,subject,region):
    all_elec=[]
    for elec in glob.glob(data_dir+'/'+subject+'/'+region+'/*'):
        all_elec.append(re.sub(data_dir+'/'+subject+'/'+region+'/','',elec))
    return all_elec

def getChannel():
    findpt_kwargs = {'filter_fn':nonshape.bandpass_default,
                'filter_kwargs': {'w':3}}
    define_true_oscillating_periods_kwargs = {'ampdiff_th':.5, 'timediff_th':.6}
    df_P, df_T = shape.compute_shape_by_cycle(x, f_range, Fs,
                                          findpt_kwargs=findpt_kwargs,
                                          define_true_oscillating_periods_kwargs=define_true_oscillating_periods_kwargs)
    return df_T
def cutoff(f,df):
    df_T_filtered=df[df['amp_mean']>f][['rdsym_time', 'sample_lastE', 'sample_nextE']]
    df_T_filtered.reset_index(inplace=True)
    return df_T_filtered

def plot_shape(raw,filtered,df,windowL,windowR,Fs):
    trial_starts = df['move_start'].values
    N_trials = len(trial_starts)
    samps_window_lim = (windowL, windowR)
    N_samps = samps_window_lim[1] - samps_window_lim[0]
    rdsym_time_ts2 = np.ones(len(raw))*np.nan
    cycle_num = len(filtered)
    for i in range(cycle_num):
        rdsym_time_ts2[filtered['sample_lastE'][i]:filtered['sample_nextE'][i]] = filtered['rdsym_time'][i]
        rdsym_time2 = np.zeros((N_trials,N_samps))
    for i, t in enumerate(trial_starts):
        rdsym_time2[i] = rdsym_time_ts2[t+samps_window_lim[0]:t+samps_window_lim[1]]
        avg_rdsym_time2 = np.nanmean(rdsym_time2,axis=0)
        sem_rdsym_time2 = sp.stats.sem(rdsym_time2,axis=0)

    t = np.arange(samps_window_lim[0]/Fs,samps_window_lim[1]/Fs,1/Fs)
    plt.plot(t,avg_rdsym_time2,'k-')
    plt.plot(t,avg_rdsym_time2-sem_rdsym_time2,'k--')
    plt.plot(t,avg_rdsym_time2+sem_rdsym_time2,'k--')
