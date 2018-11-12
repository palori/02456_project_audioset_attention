import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline #notebook
from IPython.display import display
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import os
import h5py

#### Get all files of this path
def get_dir_files(path, just_filename=True, print_info=False):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        if just_filename:
            files.extend([file for file in filenames])
        else:
            direc = dirpath.split('/')[-1]
            files.extend([direc+'/'+file for file in filenames])

    if print_info:
        print('get_dir_files\nFiles: ',files)
    return files


#### Read all files. For each file, maybe print its: name, header and shape
def read_csv_files(path, files, print_info=False):
    data = {}
    for f in files:
        file_name = path+f
        meta_data = pd.read_csv(file_name,delimiter='\t',header=0)

        # Convert to numpy arrays and strings must be 'bytes'
        for k in meta_data.keys():
            if type(meta_data[k][0]) == str:
                meta_data[k] = np.array(meta_data[k], dtype='bytes')
            else:
                meta_data[k] = np.float32(meta_data[k])

        meta_data['video_id'] = get_video_ids(data_list=meta_data['filename'])
        data[f] = meta_data
        if print_info:
            print('\n*************************\n'+f)
            print(meta_data.head())
            print(meta_data.shape)
    return data
        

#### Add a column with the video ID's in a np.array for each file
def get_video_ids(data_list, print_info=False):
    # All YouTuBe video ID's consist of 11 characters, never consider the first 'Y'
    return np.array([md[1:12] for md in data_list], dtype='bytes') # IMP! string type in 'bytes'


#### Detect files with strongly labeled data and concatenate them in a dict
def get_strongly_labeled(data, print_info=False):
    str_files = []
    for f in data.keys():
        for col in data[f].keys():
            if col == 'onset': # this file has strongly labeled data
                str_files.extend([f])
                break
    # initialization (needed to concatenate)
    str_lab = {'filename': np.array(['0'],dtype='bytes'),
                'onset': np.float32([0]),
                'offset': np.float32([0]),
                'event_label': np.array(['0'],dtype='bytes'),
                'video_id': np.array(['0'],dtype='bytes')
              }
    
    # Concatenate the strongly labeled in 'str_lab'
    for f in str_files:
        isFirst = False
        if len(str_lab['onset'])<2 and str_lab['onset']==0.0:
            isFirst = True
        
        for k in str_lab.keys():
            str_lab[k] = np.concatenate((str_lab[k], data[f][k]))
            
            if isFirst: # delete the initial value
                str_lab[k] = str_lab[k][1:]

    if print_info:
        print('\n\nStrongly labeled data files: ', str_files)
        for k in str_lab.keys():
            print('    ' + k + ': ' + str(str_lab[k].shape))

    return str_lab

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
    # pd_data = pd.DataFrame()
    # pd_data['x'] = np.array([])
    # pd_data['y'] = np.array([])
    # pd_data['id'] = np.array([])
    # pd_data.set_index('id',inplace=True)

    for f in files:
        data[f] = {}
        hdf5_path = path+f
        print('\nReading data from: '+hdf5_path)
        
        (x, y, video_id_list) = load_data(hdf5_path)
        x = np.array(uint8_to_float32(x))     # shape: (N, 10, 128)
        y = np.array(bool_to_float32(y))      # shape: (N, 527)
        video_id_list = np.array([str(v)[2:-1] for v in video_id_list], dtype='bytes') # take just the ID

        # Save in a dict
        data[f]['x'] = x
        data[f]['y'] = y
        data[f]['video_id'] = video_id_list

        # Save in a pd
        # pd_data['x'] = np.concatenate((pd_data['x'],x), axis=None)
        # pd_data['y'] = np.concatenate((pd_data['y'],y), axis=None)
        # pd_data['id'] = np.concatenate((np.array([]),video_id_list), axis=None)
        
        # print info
        if print_info:
            print('x',data[f]['x'].shape)
            print('y',data[f]['y'].shape)
            print('video id',data[f]['video_id'].shape)
    return data #{'dict_data':data, 'pd_data':pd_data}
    

###################
###################
def gen_data():

    #### START - GET DCASE LABELS ####
    dcase_path = '../dcase2018_baseline/task4/dataset/metadata/'
    dcase_files = get_dir_files(path=dcase_path, just_filename=False)

    # Read data and add column with video_id from video name
    dcase_data = read_csv_files(path=dcase_path, files=dcase_files, print_info=True)
    
    # get only strongly labeled data in a dict
    dcase_str_lab_data = get_strongly_labeled(data=dcase_data, print_info=True)

    #### END - GET DCASE LABELS ####


    #### START - GET AUDIOSET DATA ####
    audioset_path = '../packed_features/'
    audioset_files = get_dir_files(path=audioset_path, just_filename=True, print_info=False)
    # Keep just the ones with extension .h5 and NOT unbalanced data!
    audioset_files = [f for f in audioset_files if (f.split('.')[-1] == 'h5') and (f.split('_')[0] != 'unbal')]

    audioset_data = read_h5_files(path=audioset_path, files=audioset_files, print_info=True) # Still to modify

    # Let's have a look at some of the data
    """
    for d in data.keys():
        print('\n\n'+d+':')
        params = ['x','y','video_id_list']
        for p in params:
            print('\n'+p+':')
            if p == 'video_id_list':
                pass
                print(data[d][p][0])
            else:
                print(data[d][p])
    """

    #### END - GET AUDIOSET DATA ####



    #### START - SELECTING STRONGLY LABELED DATA ####
    
    print('\n\nDCASE strongly labeled data: ', dcase_str_lab_data.keys())
    print('Audioset data: ', audioset_data['bal_train.h5']['y'][0:1].shape)

    x_shape = audioset_data['bal_train.h5']['x'][0:1].shape
    y_shape = audioset_data['bal_train.h5']['y'][0:1].shape

    # initialization
    dataset = {'x': np.zeros(x_shape),
    		   'y': np.zeros(y_shape),
    		   'video_id': np.array(['0'], dtype='bytes'),
    		   'onset': np.float32([0.0]),
    		   'offset': np.float32([0.0]),
    		   'event_label': np.array(['0'], dtype='bytes'),
    		   'filename': np.array(['0'], dtype='bytes')}
               
    print('\n\nDataset', dataset)
    """
    for 
    found = False
    for ka in audioset_data.keys(): # AudioSet
		find_id = '9OqtuFGCCR8'#'Sb0169-lqLs'
		#print(v.)
		count = 0
		for id_a in audioset_data[ka]['id']: # bal_train and eval
		    if id_a == find_id:
		        print(id_a)
		        found = True

		        dataset['x'] = np.concatenate(dataset['x'], [audioset_data[ka]['x'][count]])
		        break
		    count = count + 1
		if found:
			break
    """
    #### END - SELECTING STRONGLY LABELED DATA ####

    return dataset

if __name__ == '__main__':
    dataset = gen_data()
