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


def read_csv_files(path, files, print_info=False, extract_video_id=False):
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

        if extract_video_id:
            meta_data['video_id'] = get_video_ids(data_list=meta_data['filename'])
        data[f] = meta_data
        if print_info:
            print('\n*************************\n'+f)
            print(meta_data.head(10))
            print(meta_data.shape)
    return data

if __name__ == '__main__':
    path = '../dcase2018_baseline/task4/dataset/metadata/test/'
    files = ['test.csv']

    read_csv_files(path, files, print_info=True)


    """
    DCASE - 10 classes       ->     AudioSet Equivalent     Code
    ------------------              -------------------     ----
    Vacuum_cleaner                  Vacuum cleaner                      /m/0d31p
    Frying                          Frying (food)                       /m/0dxrf
    Cat                             Cat                                 /m/01yrx
    Alarm_bell_ringing              -- spread in different classes --
    Running_water                   -- spread in different classes --
    Speech                          Speech                              /m/09x0r (also spread but took the general one)
    Electric_shaver_toothbrush      Electric toothbrush                 /m/04fgwm
                                    Electric shaver, electric razor     /m/02g901
    Blender                         Blender                             /m/02pjr4
    Dishes                          Dishes, pots, and pans              /m/04brg2
    Dog                             Dog                                 /m/0bt9lr (also spread but took the general one)
    """

