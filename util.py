import glob
import re
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
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

def plot_shape(raw,filtered,df,windowL,windowR,Fs,subj,electrode):
    trial_starts = df['move_start'].astype(int).values
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
    #plt.figure(figsize=(12,4))
    #plt.plot(t,avg_rdsym_time2,'k-')
    #plt.plot(t,avg_rdsym_time2-sem_rdsym_time2,'k--')
    #plt.plot(t,avg_rdsym_time2+sem_rdsym_time2,'k--')
    #plt.savefig("C:/Users/Yimeng/Documents/GitHub/LabWork/plots/"+subj+electrode+".png")
    return rdsym_time2,avg_rdsym_time2,sem_rdsym_time2

def drop_bad_trials(df,threshold,column,less=True,drop_other=True):
    df_no_bad_trials = df.dropna()
    if(drop_other):
        if(less):
            df_no_bad_trials = df_no_bad_trials[df_no_bad_trials[column]<threshold]
        else:
            df_no_bad_trials = df_no_bad_trials[df_no_bad_trials[column]>threshold]
    return df_no_bad_trials

# Functions for generating the dataframe for behavioral data
def generateDataFrame(cue_diff,fingers, subj):
    data_frame={}

    #Find cue start and end for each trial
    data_frame['cue_start']=np.where(cue_diff>0)[0]
    data_frame['cue_end']=np.where(cue_diff<0)[0]

    if subj == 'jp':
        data_frame['cue_end'] = np.delete(data_frame['cue_end'],0)
    elif subj == 'wm':
        data_frame['cue_start'] = np.delete(data_frame['cue_start'],-1)

    #Find corresponding fingers for each trial
    data_frame['finger'] = np.array(cue_diff[data_frame['cue_start']])

    data_frame['move_start'] = []
    data_frame['move_end'] = []
    for i in data_frame['cue_start']:
        if sum(fingers['start'][cue_diff[i]-1]>i) >= 1:
            data_frame['move_start'].append(fingers['start'][cue_diff[i]-1][fingers['start'][cue_diff[i]-1]>i][0])
        else:
            data_frame['move_start'].append(np.nan)
    for i in data_frame['cue_end']:
        if sum(fingers['end'][abs(cue_diff[i])-1]>i) >= 1:
            data_frame['move_end'].append(fingers['end'][abs(cue_diff[i])-1][fingers['end'][abs(cue_diff[i])-1]>i][0])
        else:
            data_frame['move_end'].append(np.nan)
    data_frame['move_start'] =  np.array(data_frame['move_start'])
    data_frame['move_end'] = np.array(data_frame['move_end'])
    data_frame['delay_start']=data_frame['move_start']-data_frame['cue_start']
    data_frame['delay_end']=data_frame['move_end']-data_frame['cue_end']
    data_frame['duration']=data_frame['move_end']-data_frame['move_start']
    return data_frame

def generateDataframeAll(all_cues_diff,all_subj_fingers):
    dfs = []
    data_frame_all=[]
    for i in all_cues_diff.keys():
        data_frame=generateDataFrame(all_cues_diff[i],all_subj_fingers[i], i)
        dfs.append(pd.DataFrame.from_dict(data_frame))
        data_frame_all.append(data_frame)
        dfs[-1]['subject'] = i

    return dfs,data_frame_all

def getAllSubjects(data_dir, subjects,subj_name=2):
    all_subjects = []
    # Get all subject files
    for sub in subjects:
        # Get
        if len(sub) == len(data_dir)+1+subj_name:
            file_name = sub[-subj_name:]
            all_subjects.append(file_name)
    return all_subjects

def loadCue(all_subjects,data_dir):
    all_cues_diff={}
    # Load cue for each subject and find start/stop times
    for subjcode in all_subjects:

        cue = np.load(data_dir+'/'+subjcode+'/cue.npy')
        all_cues_diff[subjcode]=np.diff(cue)
    return all_cues_diff

def getAllMovement(all_subjects,data_dir):
    all_subj_fingers={}

    for subjcode in all_subjects:
        # Compute finger start and stop times
        all_subj_fingers[subjcode]={'start':{},'end':{}}
        raw_finger=[]
        for i in range(5):
            finger = np.load(data_dir+'/'+subjcode+'/finger'+str(i)+'.npy')
            raw_finger.append(finger)

            all_subj_fingers[subjcode]['start'][i],all_subj_fingers[subjcode]['end'][i]=fingerStartEnd(finger)
    return all_subj_fingers

def getAllCue(all_subjects,all_cues_diff):
    all_cues={}
    for subjcode in all_subjects:
        all_cues[subjcode]={'start':{},'end':{}}
        for index in range(5):
            all_cues[subjcode]['start'][index] = np.where(all_cues_diff==index+1)[0]
            all_cues[subjcode]['end'][index] = np.where(all_cues_diff==-index-1)[0]
    return all_cues

def plotCueAndFinger(all_subjects,raw_finger,all_subj_fingers,all_cues):
    for subjcode in all_subjects:
        plt.figure(figsize=(16,8))

        f, sub = plt.subplots(5, sharex=True,figsize=(16,8))
        for k in range(5):
            sub[k].plot(raw_finger[k])

            for start,end in zip(all_subj_fingers[subjcode]['start'][k],all_subj_fingers[subjcode]['end'][k]):
                sub[k].axvline(x=start, ymin=0, ymax = 3000, linewidth=2, color='k')
                sub[k].axvline(x=end, ymin=0, ymax = 3000, linewidth=2, color='r', alpha=0.5)
            for cueS, cueE in zip(all_cues[subjcode]['start'][k],all_cues[subjcode]['end'][k]):
                sub[k].axvline(x=cueS, ymin=0, ymax = 3000, linewidth=2, color='g')
                sub[k].axvline(x=cueE, ymin=0, ymax = 3000, linewidth=2, color='y')
            plt.xlim(0,50000)
# Delete movements unrelated to local cue
def deleteStartEnd(fingers,interval):
    for k in range(5):
        for i, j in zip(fingers['start'][k],fingers['end'][k]):
            if (j-i<interval):
                fingers['start'][k].remove(i)
                fingers['end'][k].remove(j)
# Find start and end times of movement
def fingerStartEnd (finger, thresh = 50, time_thresh = 1000):
    all_start_samps = []
    all_end_samps = []

    finger_deriv = np.abs(np.diff(finger))
    is_moving = finger_deriv > thresh
    move_samples = np.argwhere(is_moving==1).transpose()[0]
    move_samples_diff = np.diff(move_samples)

    all_start_samps.append(move_samples[0])

    N_samps = len(move_samples)
    for i in range(1,N_samps):
        if move_samples_diff[i-1] > time_thresh:
            all_start_samps.append(move_samples[i])
            all_end_samps.append(move_samples[i-1])
    all_end_samps.append(move_samples[-1])
    return np.array(all_start_samps), np.array(all_end_samps)
def calculate_average(trial,label):
    total = 0
    if(len(trial)!=0):
        for num in trial[label]:
            total += num;
        return total/len(trial)
    return 0
def dictionary_to_csv(dictionary,name):
    for subj in dictionary.keys():
        dictionary[subj].to_csv(name+subj+'.csv')
