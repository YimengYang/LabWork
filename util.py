import glob
import re
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
