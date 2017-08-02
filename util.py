import glob
import re
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import Process



def load_paths(data_dir='C:/data2/dg/'):
    """ Given the directory path, load the paths to each eletrode of different
    regions

    Parameter:
        data_dir: The directory where all signals are (default is 'C:/data2/dg/')

    Return:
        data_paths: A dictionary contains all paths, with subject as key

        all_regions: A dictioanry contain all regions associated with each subject
    """
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
    """ Get a list of subjects given the directory and the length of the name
    of each subject

    Parameter:
        data_dir: The directory where all data stored

        name_len: The length of the subject name code (default is 2)

    Return:
        all_subjects: A list of all subjects names
    """

    all_subjects=[]
    subjects=glob.glob(data_dir+'/*')
    for sub in subjects:
        if len(sub) == len(data_dir)+name_len:
            all_subjects.append(sub[-name_len:])
    return all_subjects

def _get_region(data_dir,subject):

    """ Get all regions related to the subject given the directory

    Parameter:
        data_dir: The directory where all data stored

        subject: The subject name code

    Return:
        all_reg: A list of all subjects names
    """

    all_reg=[]

    # Loop through all folder under the subject folder
    for reg in glob.glob(data_dir+'/'+subject+'/*'):
        if reg[-1:].isdigit():
            all_reg.append(reg[-1:])
    return all_reg

def _get_electrode(data_dir,subject,region):
    """ Get all electrode of a certain region of a certain subject given the
    directory

    Parameter:
        data_dir (String): The directory where all data stored

        subject (String): The subject name code

        region (String): The region that contains the electrode we need

    Return:
        all_elec (list): A list of all eletrode numbers
    """

    all_elec=[]
    for elec in glob.glob(data_dir+'/'+subject+'/'+region+'/*'):
        all_elec.append(re.sub(data_dir+'/'+subject+'/'+region+'/','',elec))
    return all_elec

def cutoff(f,df):
    """ Help cut off some data where amp_mean is greater than a certain threshold

    Parameter:
        f: The threshold

        df: The dataframe that we need to operate on

    Return:
        df_T_filtered: The dataframe after the cutoff
    """

    df_T_filtered=df[df['amp_mean']>f][['rdsym_time', 'sample_lastE', 'sample_nextE']]
    df_T_filtered.reset_index(inplace=True)
    return df_T_filtered

def plot_shape(raw, filtered, df, windowL, windowR, Fs, subj, electrode,
                plot=True):
    """ For plotting the diagrams for features of rdsym
    """
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
    if plot:
        plt.figure(figsize=(12,4))
        plt.plot(t,avg_rdsym_time2,'k-')
        plt.plot(t,avg_rdsym_time2-sem_rdsym_time2,'k--')
        plt.plot(t,avg_rdsym_time2+sem_rdsym_time2,'k--')
        plt.savefig("C:/Users/Yimeng/Documents/GitHub/LabWork/plots/"+subj+electrode+".png")
    return rdsym_time2,avg_rdsym_time2,sem_rdsym_time2

def drop_bad_trials(df, threshold, column, less=True, drop_other=True):
    """ Drop nan and bad trials that doesn't meet the threshold in the dataframe

    Parameter:
        df (pandas dataframe): The dataframe that we need to operate on

        threshold: The threshold of the data

        column (String): The certain column that we want to check the threshold

        less (boolean): Check if the data is less than or greater than the
                        threshold (default True)

        drop_other (boolean): Check if need to check things other than nan

    Return:
        df_no_bad_trials (pandas dataframe): The dataframe with good trials
    """
    df_no_bad_trials = df.dropna()

    if drop_other:
        if less:
            df_no_bad_trials = df_no_bad_trials[df_no_bad_trials[column]<threshold]
        else:
            df_no_bad_trials = df_no_bad_trials[df_no_bad_trials[column]>threshold]
    return df_no_bad_trials


def generate_dataframe(cue_diff, fingers, subj):
    """ Generate the dataframe for behavioral data

    Parameter:
        cue_diff: The cue data

        fingers: The movement data

        subj: The name of the subject

    Return:
        data_frame: The result dataframe
    """
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

    # Extract the move_start and move_end based on the cue_start and cue_end
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

    # Convert the columns to np array and calculate some other columns
    data_frame['move_start'] =  np.array(data_frame['move_start'])
    data_frame['move_end'] = np.array(data_frame['move_end'])
    data_frame['delay_start']=data_frame['move_start']-data_frame['cue_start']
    data_frame['delay_end']=data_frame['move_end']-data_frame['cue_end']
    data_frame['duration']=data_frame['move_end']-data_frame['move_start']
    return data_frame

def generate_dataframe_all(all_cues_diff, all_subj_fingers):
    """ Generate a master dataframe given all cues and finger data

    Parameter:
        all_cues_diff: all cue data for all subjects

        all_subj_fingers: all finger data for all subjects

    Return:
        dfs (list of pandas datarame): The master dataframe

        data_frame_all (list of dictionary): The master dataframe
    """
    dfs = []
    data_frame_all=[]
    for i in all_cues_diff.keys():
        data_frame=generate_dataframe(all_cues_diff[i],all_subj_fingers[i], i)
        dfs.append(pd.DataFrame.from_dict(data_frame))
        data_frame_all.append(data_frame)
        dfs[-1]['subject'] = i

    return dfs,data_frame_all

def get_subjects_all(data_dir, subj_name=2):
    """ Get the list of subject names given directory,
    """
    all_subjects = []
    subjects=glob.glob(data_dir+'/*')
    # Get all subject files
    for sub in subjects:
        # Get
        if len(sub) == len(data_dir)+1+subj_name:
            file_name = sub[-subj_name:]
            all_subjects.append(file_name)
    return all_subjects

def load_raw_finger(data_dir, all_subjects):
    """ Load the raw finger data for each
    """
    raw_finger_all={}
    for subjcode in all_subjects:
        raw_finger = []
        for i in range(5):
            finger = np.load(data_dir+'/'+subjcode+'/finger'+str(i)+'.npy')
            raw_finger.append(finger)
        raw_finger_all[subjcode]=raw_finger
    return raw_finger_all
def load_cue(all_subjects,data_dir):
    """ Load all the cues given all subjects and the data directory

    Parameter:
        all_subject (list of String): A list of all subjects name

        data_dir (String): The data directory

    Return:
        all_cues_diff (dicionary): A dictionary of all processed cue data
    """

    all_cues_diff={}

    # Load cue for each subject and find start/stop times
    for subjcode in all_subjects:
        cue = np.load(data_dir+'/'+subjcode+'/cue.npy')
        all_cues_diff[subjcode]=np.diff(cue)

    return all_cues_diff

def get_movement_all(all_subjects,data_dir):
    """ Load all the movement data given all subjects and the data directory

    Parameter:
        all_subject (list of String): A list of all subjects name

        data_dir (String): The data directory

    Return:
        all_subj_fingers (dicionary): A dictionary of all processed cue data
    """
    all_subj_fingers={}

    for subjcode in all_subjects:
        # Compute finger start and stop times
        all_subj_fingers[subjcode]={'start':{},'end':{}}
        raw_finger=[]
        for i in range(5):
            finger = np.load(data_dir+'/'+subjcode+'/finger'+str(i)+'.npy')
            raw_finger.append(finger)

            all_subj_fingers[subjcode]['start'][i],all_subj_fingers[subjcode]['end'][i]=find_finger_start_end(finger)
    return all_subj_fingers

def get_cue_all(all_subjects,all_cues_diff):
    """ Get all cue start and end data given the subjects and processed cue data

    Parameter:
        all_subject (list of String): A list of all subjects name

        all_cues_diff (dicionary): A dictionary of all processed cue data

    Returns:
        all_cues (dictionary): All cue data with start and end for each trial
    """

    all_cues={}
    for subjcode in all_subjects:
        all_cues[subjcode]={'start':{},'end':{}}

        # For each finger load all cue start and end time for that finger
        for index in range(5):
            all_cues[subjcode]['start'][index] = np.where(all_cues_diff==index+1)[0]
            all_cues[subjcode]['end'][index] = np.where(all_cues_diff==-index-1)[0]
    return all_cues

def plot_cue_and_finger(all_subjects, raw_finger, all_subj_fingers, all_cues):
    """ Plot cue and finger data together for each subject

    Parameter:
        all_subject (list of String): A list of all subjects name

        raw_finger: The raw data for finger movement

        all_subj_fingers: The processed finger data for each subject

        all_cues (dictionary): All cue data with start and end for each trial
    """

    # Loop through each subject
    for subjcode in all_subjects:
        plt.figure(figsize=(16,8))

        f, sub = plt.subplots(5, sharex=True,figsize=(16,8))
        for k in range(5):
            sub[k].plot(raw_finger[subjcode][k])

            for start,end in zip(all_subj_fingers[subjcode]['start'][k],all_subj_fingers[subjcode]['end'][k]):
                sub[k].axvline(x=start, ymin=0, ymax = 3000, linewidth=2, color='k')
                sub[k].axvline(x=end, ymin=0, ymax = 3000, linewidth=2, color='r', alpha=0.5)
            for cueS, cueE in zip(all_cues[subjcode]['start'][k],all_cues[subjcode]['end'][k]):
                sub[k].axvline(x=cueS, ymin=0, ymax = 3000, linewidth=2, color='g')
                sub[k].axvline(x=cueE, ymin=0, ymax = 3000, linewidth=2, color='y')
            plt.xlim(0,50000)


def delete_start_end(fingers, interval):
    """ Delete movements unrelated to local cue

    Parameter:
        fingers: Movement data with start and end times

        interval: The interval that we decide it is not appropriate

    Return:
        fingers: Bad start and end points deleted
    """
    for k in range(5):
        for i, j in zip(fingers['start'][k],fingers['end'][k]):
            if (j - i < interval):
                fingers['start'][k].remove(i)
                fingers['end'][k].remove(j)
    return fingers


def find_finger_start_end (finger, thresh = 50, time_thresh = 1000):
    """ Find start and end times of movement

    Parameter:
        finger: The finger movement data

        thresh: The threshold for determining the whether finger is moving

        time_thresh: The threshold for determining the start and end points

    Return:
        np array of all start points
        np array of all end points
    """

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

def calculate_average(trial, column):
    """ Calulate the average of a certain column given the trials

    Parameter:
        trial (list): The list of trial that need to be averaged

        column (String): The column the we want to average the data_dir

    Return:
        The average
    """
    total = 0
    if(len(trial)!=0):
        for num in trial[column]:
            total += num;
        return total/len(trial)
    return 0

def parse_electrode(paths):
    """ Parse the electrode number given the path to it

    Parameter:
        path (String): The path to the electrode

    Return:
        The electrode number
    """

    electrode=paths.split('\\')[1]
    return electrode[:-4]

def calculate_rdsym_average(behavior_move_all,data_paths,Fs=1000,f_range=(10,30),interval=500,move=True):
    """ Calculate the average of rdsym for all subjects
    """
    tasks=[]
    e=ThreadPoolExecutor()
    for bmove_each in behavior_move_all.values():
        result = e.submit(calculate_rdsym_average_subj,bmove_each,data_paths,Fs,f_range,interval,move)
        tasks.append(result)
    for task in tasks:
        task.result()

def calculate_rdsym_average_subj(bmove_each,data_paths,Fs,f_range,interval,move):
    subj = bmove_each['subject'][0]
    move_start=bmove_each['move_start']
    move_end=bmove_each['move_end']
    dfneural_all_subj=pd.DataFrame({'start_time':move_start,'end_time':move_end})

    for region in data_paths[subj]:
        for path in data_paths[subj][region]:
            rdsym=[]
            electrode = parse_electrode(path)
            signal = np.load(path) # voltage series

            signal = Process.preprocess_filters(signal,Fs)
            df_electrode = shape.features_by_cycle(signal, Fs, f_range, center_extrema='T' )
            df_electrode['subject']=subj
            df_electrode.dropna()
            for index in range(len(move_start)):
                if move:
                    trial = df_electrode[(df_electrode['sample_trough']<=move_end[index]) & (df_electrode['sample_trough']>=move_start[index])]
                    avg=calculate_average(trial,'time_rdsym')
                else:
                     if index != len(move_start)-1:
                        if move_start[index+1]-move_end[index]<2*interval:
                            avg=np.nan
                        else:
                            middle = (move_end[index]+move_start[index+1])/2
                            trial= df_electrode[(df_electrode['sample_trough']>=middle-interval) & (df_electrode['sample_trough']<=middle+interval)]
                            avg=calculate_average(trial,'time_rdsym')
                rdsym.append(avg)
            rdsym_series=pd.Series(rdsym)

            dfneural_all_subj['rdsym_elec'+electrode]=rdsym_series.values
    dfneural_all_subj.reindex_axis(sorted(dfneural_all_subj.columns), axis=1)
    if move:
        dfneural_all_subj.to_csv('./neural_move_'+subj+'.csv')
    else:
        dfneural_all_subj.to_csv('./neural_nomove_'+subj+'.csv')
