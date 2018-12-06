# test_data_gen.py

import data_generator as dg
import h5py
import numpy as np

#### Fast reduction, maybe it become unbalanced
def reduce_dataset(path, save_path, file_name, init_file=0, end_file=3000, save=True):
    print('\n\n---\n---')
    data = dg.read_h5_files(path, [file_name], True)

    print('\n---')

    if save:
        file = save_path+file_name
        f = h5py.File(file, 'a') # read/create the file

    for k in data.keys():
        print(k)
        for k1 in data[k].keys():
            print('  ',k1)
            if save:
                f.create_dataset(k1, data=data[k][k1][init_file:end_file]) # selecting just the first 'num_files'


if __name__ == '__main__':
    path = '../packed_features/'
    save_path = 'test_data/recised_3000_6000/'
    reduce_dataset(path, save_path, 'bal_train.h5', 3000, 6000)
    reduce_dataset(path, save_path, 'eval.h5', 3000, 6000)
    save_path = 'test_data/recised_0_6000/'
    reduce_dataset(path, save_path, 'bal_train.h5', 0, 6000)
    reduce_dataset(path, save_path, 'eval.h5', 0, 6000)


 
""" #### INCLUDED IN 'data_generator.py'

# Create the file and fill it
filename = 'strongly_labeled_data.hdf5'
f = h5py.File(filename, 'a') # read/create the file
filekeys = [k for k in f.keys()]

if len(filekeys)<1: # is an empty file, just created
    #need to generate the data and save to the file
    arr = np.arange(100)
    print('arr = ',arr)
    dset = f.create_dataset("init", data=arr)

    # genetate the data from other files and combine in a dict
    dataset = gen_data()
    grp = f.create_group('grp')
    for k in dataset.keys(): # convert from dict to hdf5
    	'''
    	IMP! data preferably as np.array with a specified 'dtype'
    	Eg: for 'strings' (video id,) dataset[k] could be
    	    np.array(['9OqtuFGCCR8','Sb0169-lqLs'], dtype='bytes')
    	'''
    	grp.create_dataset(k, data=dataset[k]) 

# Load the file
data = f.get('init')
data = np.array(data)
print('\n\ndata = ',data)

"""