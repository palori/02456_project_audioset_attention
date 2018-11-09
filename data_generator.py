import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#notebook
from IPython.display import display
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import os
import h5py

#### Get all files of this path
def get_dir_files(path, just_filename=True):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        if just_filename:
            files.extend([file for file in filenames])
        else:
            direc = dirpath.split('/')[-1]
            files.extend([direc+'/'+file for file in filenames])
    print('get_dir_files\nFiles: ',files)
    return files


#### Read all files. For each file, maybe print its: name, header and shape
def read_csv_files(path, files, print_info=False):
    data = {}
    for f in files:
        file_name = path+f
        meta_data = pd.read_csv(file_name,delimiter='\t',header=0)
        data[f] = meta_data
        if print_info:
            print('\n*************************\n'+f)
            print(meta_data.head())
            print(meta_data.shape)
    return data
        

# Saving the video ID's in a np.array for each file
def get_video_ids(data):
    video_ids = {}
    for k in data.keys():
        # All YouTuBe video ID's consist of 11 characters, never consider the first 'Y'
        video_ids[k] = np.array([md[1:12] for md in data[k]['filename']])
    return video_ids


#######################
#######################

#### How they do it in the Readme file ####
def load_data(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        video_id_list = hf.get('video_id_list')
        x = np.array(x)
        y = list(y)
        video_id_list = list(video_id_list)
        
    return x, y, video_id_list

#### From Readme ####
def uint8_to_float32(x):
    return (np.float32(x) - 128.) / 128.

#### From Readme ####
def bool_to_float32(y):
    return np.float32(y)

#### Read all 'h5' files
def read_h5_files(path, files, print_info=False): # Still to modify
    #initialization of outputs
    data = {}
    pd_data = pd.DataFrame()
    pd_data['x'] = np.array([])
    pd_data['y'] = np.array([])
    pd_data['id'] = np.array([])
    pd_data.set_index('id',inplace=True)

    for f in files:
        data[f] = {}
        hdf5_path = path+f
        print('\nReading data from: '+hdf5_path)
        
        (x, y, video_id_list) = load_data(hdf5_path)
        x = np.array(uint8_to_float32(x))     # shape: (N, 10, 128)
        y = np.array(bool_to_float32(y))      # shape: (N, 527)
        video_id_list = np.array([str(v)[2:-1] for v in video_id_list]) # take just the ID

        data[f]['x'] = x
        data[f]['y'] = y
        data[f]['video_id_list'] = video_id_list

        pd_data['x'] = np.concatenate((pd_data['x'],x), axis=None)
        pd_data['y'] = np.concatenate((pd_data['y'],y), axis=None)
        pd_data['id'] = np.concatenate((np.array([]),video_id_list), axis=None)
        
        # print info
        if print_info:
            print('x',x.shape)
            print('y',y.shape)
            print('video id',len(video_id_list))
    return {'dict_data':data, 'pd_data':pd_data}
    

###################
###################
def gen_data():

    #### START - GET DCASE LABELS ####
    dcase_path = '../dcase2018_baseline/task4/dataset/metadata/'
    dcase_files = get_dir_files(path=dcase_path, just_filename=False)
    data = read_csv_files(path=dcase_path, files=dcase_files, print_info=False)
    #print(data.keys())
    video_ids = get_video_ids(data=data)
    #print(video_ids.keys())

    # Check that it worked for the first 2 videos in the first file.
    #print('\n\nClip names:\n',data[list(data.keys())[0]]['filename'][:2])
    #print('\n\nClip ids:\n',list(video_ids.values())[0][:2])

    dcase_eval = data['eval/eval.csv']
    #print(dcase_eval.head())
    dcase_eval._set_item('id', video_ids['eval/eval.csv'])
    dcase_eval.set_index('id',inplace=True)
    #print(dcase_eval.head())

    #### END - GET DCASE LABELS ####



    #### START - GET AUDIOSET DATA ####
    audioset_path = '../packed_features/'
    audioset_files = get_dir_files(path=audioset_path, just_filename=True)
    # Keep just the ones with extension .h5 and NOT unbalanced data!
    audioset_files = [f for f in audioset_files if (f.split('.')[-1] == 'h5') and (f.split('_')[0] != 'unbal')]

    audioset_data = read_h5_files(path, files, print_info=False) # Still to modify
    # because we want it in DataFrame and not as dict, see the function
    audioset_data = audioset_data['pd_data']

    # Let's have a look at some of the data
    """
    for d in data.keys():
        print('\n\n'+d+':')
        params = ['x','y','video_id_list']
        for p in params:
            print('\n'+p+':')
            print(data[d][p][0])
    """

    #### END - GET AUDIOSET DATA ####



    #### START - SELECTING STRONGLY LABELED DATA ####
    dataset = pd.merge(dcase_eval, audioset_data, on='id', how='inner') # inner = onñy if it exixt in both
    print()

    #### END - SELECTING STRONGLY LABELED DATA ####

if __name__ == '__main__':
    gen_data()