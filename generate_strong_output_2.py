# test_data_gen.py

import data_generator as dg
import h5py
import numpy as np
import pandas as pd

# Gererate new output 'y' with less classes and
# strongly labeled output 'ys'

#### Add a column with the video ID's in a np.array for each file
def get_video_ids(data_list, print_info=False):
    # All YouTuBe video ID's consist of 11 characters, never consider the first 'Y'
    return np.array([md[1:12] for md in data_list], dtype='bytes') # IMP! string type in 'bytes'


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
            print(meta_data.head(10))
            print(meta_data.shape)
    return data, meta_data


def create_out():
    path = '../dcase2018_baseline/task4/dataset/metadata/eval/'
    files = ['eval.csv']

    data, meta_data = read_csv_files(path, files, print_info=False)
    meta_data = meta_data.dropna()
    label = meta_data.drop_duplicates('event_label')
    ids = meta_data.drop_duplicates('video_id')
    label = label.sort_values(by=['event_label'])
    print(meta_data.head())
    for i in range(0,len(meta_data.index)):
        meta_data.at[i,'onset'] = int(meta_data.iloc[i]['onset'])
        meta_data.at[i,'offset'] =  round(meta_data.iloc[i]['offset'])
    
    T = 10
    out = np.zeros((len(ids.index),T,len(label.index)))
    labelv=label.values[:,[3]]
    idsv = ids.values[:,[4]]
    print(meta_data.head())
    for i in range(0,len(ids.index)):
        for k in range(0,len(meta_data.index)):
            if idsv[i] == meta_data.iloc[k]['video_id']:
                for l in range(0,len(label.index)):
                    if labelv[l] == meta_data.iloc[k]['event_label']:
                        for t in range(int(meta_data.iloc[k]['onset']),int(meta_data.iloc[k]['offset'])):
                            out[i,t,l]= 1
                            
                            
    print(out[0,:,4])
    return out
    
   

    
    
    
    

    
    

if __name__ == '__main__':
    out = create_out()


