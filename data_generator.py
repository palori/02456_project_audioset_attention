import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline #notebook
from IPython.display import display
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import os
import h5py
import math

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
def read_csv_files(path, files, delimiter='\t', header=0, print_info=False, extract_video_id=False): # @@@@ new arguments
	data = {}
	for f in files:
		file_name = path+f
		meta_data = pd.read_csv(file_name,delimiter=delimiter,header=header)

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
			print(meta_data.head())
			print(meta_data.shape)
	return data, meta_data
		

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
def read_h5_files(path, files, print_info=False):
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
		data[f]['video_id_list'] = video_id_list
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


def print_dict_shape(dict_data, title):
	print('\n'+title+': ')
	for k in dict_data.keys():
		print("    "+k+":", dict_data[k].shape)

def print_audioset_shape(audioset_data):
	print('\nAudioset data:')
	for kf in audioset_data.keys():
		print('  '+kf+':')
		for k in audioset_data[kf].keys():
			try:
				print('    '+k+':', audioset_data[kf][k].shape)
			except AttributeError:
				print('    '+k+':', len(audioset_data[kf][k]))

##################
##################


def reduce_output_size(audioset_data, new_output_classes):
	m = new_output_classes

	new_data = {}

	index2drop = []
	#print("    q= ",m.iloc[37]['dcase_label'] )
	for i in range(0,len(m.index)):
		if m.iloc[i]['dcase_label'] == b'nan':#float('nan'):

			index2drop.append(i)

	m.drop(index2drop, axis=0, inplace=True)

	# print(m.head(66))

	for fa in audioset_data.keys(): # files
		real_in = audioset_data[fa]['x']
		real_out = audioset_data[fa]['y']
		real_ids = audioset_data[fa]['video_id']

		new_data[fa] = {}

		#reshape output form 527 to 10
		mapping = np.zeros((10,527)) # initialization
		for mm in range(0,len(m.index)):
			#print("    row['index']:", m.iloc[mm]['index'])
			col = int(m.iloc[mm]['index'])
			dc_labels = str(m.iloc[mm]['dcase_label']).split(',')

			for dcl in dc_labels:
				row = -1
				#print("    dcl: ",dcl, ", type: ", type(dcl))
				if dcl == "b'Alarm_bell_ringing'":
					row = 0
				elif dcl == "b'Blender'":
					row = 1
				elif dcl == "b'Cat'":
					row = 2
				elif dcl == "b'Dishes'":
					row = 3
				elif dcl == "b'Dog'":
					row = 4
				elif dcl == "b'Electric_shaver_toothbrush'":
					row = 5
				elif dcl == "b'Frying'":
					row = 6
				elif dcl == "b'Running_water'":
					row = 7
				elif dcl == "b'Speech'":
					row = 8
				elif dcl == "b'Vacuum_cleaner'":
					row = 9
				else:
					print("\n    Error, no DCASE label fot that index!\n")
				if row > -1:
					mapping[row,col] = 1

		#print("Mapping matrix=\n",mapping)
		
		
		new_in = np.empty((1,10,128))
		new_out = np.empty((1,10))
		new_ids = np.empty((1))
		

		count = 0
		for k in range(0,real_out.shape[0]):
			cmd=0
			out10 = np.matmul(mapping,np.transpose(real_out[k])) # new output
			#out10 = np.transpose(out10)
			#print("    out shape: ",out10.shape)

			for out in out10:
				if out == 1:
					new_in = np.concatenate((new_in, [real_in[k]]),axis=0)
					new_out = np.concatenate((new_out, [out10]),axis=0)
					new_ids = np.append(new_ids, real_ids[k])
					if (False):
						print("\n---- new_in:",new_in)
						print("\n---- new_out:",new_out)
						print("\n---- new_ids:",new_ids)
					if count < 1:
						new_in = np.delete(new_in,0,0)
						new_out = np.delete(new_out,0,0)
						new_ids = np.delete(new_ids,0,0)
						count = 10
						if (False):
							print("\n-**- new_in:",new_in)
							print("\n-**- new_out:",new_out)
							print("\n-**- new_ids:",new_ids)

					print("\nIteration ",k)
					print("    new_in: ", new_in.shape)
					print("    new_out: ", new_out.shape)
					print("    new_ids: ", new_ids.shape)
					#print("        out: ", out10)
					break


		#print(new_out.shape)
		new_data[fa]['x'] = new_in
		new_data[fa]['y'] = new_out
		new_data[fa]['video_id'] = new_ids
		print_audioset_shape(new_data)
		
	return new_data



##################
##################


#### Dataset 0-initialization
def dataset_init(x_shape, y_shape):
	return {'x': np.zeros(x_shape),
			'y': np.zeros(y_shape),
			'video_id': np.array(['0'], dtype='bytes'),
			'video_id_list': np.array(['0'], dtype='bytes'), #@@@@
			'onset': np.float32([0.0]),
			'offset': np.float32([0.0]),
			'event_label': np.array(['0'], dtype='bytes'),
			'filename': np.array(['0'], dtype='bytes')}

#### Find matches with 'video_id' and return the concatenation of them
def concatenate_matches(dcase_str_lab_data, audioset_data):
	# initialization
	try:
		x_shape = audioset_data['bal_train.h5']['x'][0:1].shape
		y_shape = audioset_data['bal_train.h5']['y'][0:1].shape
	except KeyError:
		try:
			x_shape = audioset_data['eval.h5']['x'][0:1].shape
			y_shape = audioset_data['eval.h5']['y'][0:1].shape
		except KeyError:
			print("    KeyError!!!")
	dataset = dataset_init(x_shape, y_shape)
			   
	#print('\n\nDataset', dataset) # 0-initialization
	
	d_index = -1 # counter for dcase to be able to concatenate when match
	for id_d in dcase_str_lab_data['video_id']: # ids DCASE
		found = False
		d_index = d_index + 1 

		for ka in audioset_data.keys(): # AudioSet files
			a_index = -1 # counter for audioset to be able to concatenate when match
			
			for id_a in audioset_data[ka]['video_id']: # ids AudioSet
				a_index = a_index + 1 
				if id_a == id_d:
					found = True
					# Concatenate values
					#             from AudioSet
					for k in audioset_data[ka].keys():
						dataset[k] = np.concatenate((dataset[k], audioset_data[ka][k][a_index:a_index+1]))	
					#             from DCASE
					for k in dcase_str_lab_data.keys():
						if k != 'video_id':
							dataset[k] = np.concatenate((dataset[k], dcase_str_lab_data[k][d_index:d_index+1]))

					break
			if found:
				break
	
	return dataset

###################
###################

def create_strong_output(path, file, print_info=False):
    data, meta_data = read_csv_files(path, file, delimiter='\t', header=0, print_info=False, extract_video_id=True)
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
                            
                            
    if print_info:
    	print(out[0,:,4])
    return idsv, out


def save2h5(f, dataset):
	for k in dataset.keys():
		print('\n  '+k+' type=', type(dataset[k][0]))
		if k != 'event_label' and k != 'filename': # @@@@temporal, need to be fixed!!!
			print('    ... saving '+k+'...')
			f.create_dataset(k, data=dataset[k])

###################
###################

def gen_data(dataset_file_name,
			 dcase_path='../dcase2018_baseline/task4/dataset/metadata/',
			 audioset_path='../packed_features/'):

	#### START - SAVE IN 'H5' FILE ####
	f = h5py.File(dataset_file_name, 'a') # read/create the file
	file_exist = False
	try:
		filekeys = [k for k in f.keys()]
		if (filekeys != []):
			print('\n\nFile "'+dataset_file_name+'" exists')
			print('\nfilekeys:',filekeys)
			file_exist = True
	except KeyError:
		print('\n\nInexistant file "'+dataset_file_name+'"')

	if file_exist: # file not empty, proceed to read it	
		dataset = dataset_init((1, 10, 128), (1, 527)) #initialization with all needed keys
		for k in f.keys():
			if k != 'a_event_label' and k != 'a_filename': # @@@@temporal, need to be fixed!!!
				print('\n    ... loading '+k+'...', type(dataset[k][0]))
				a = f.get(k)
				if type(dataset[k][0]) != np.ndarray:
					dataset[k] = np.array(a, dtype=type(dataset[k][0]))
				else:
					dataset[k] = np.array(a)

		print_dict_shape(dict_data=dataset, title='dataset shape')

	else: # is an empty file, just created
		#need to collect the data and save to the file

		#### START - GET DCASE LABELS ####
		dcase_files = get_dir_files(path=dcase_path, just_filename=False)

		# Read data and add column with video_id from video name
		dcase_data, a = read_csv_files(path=dcase_path, files=dcase_files, delimiter='\t', header=0, print_info=True, extract_video_id=True)
		
		# get only strongly labeled data in a dict
		dcase_str_lab_data = get_strongly_labeled(data=dcase_data, print_info=True)
		#### END - GET DCASE LABELS ####


		#### START - GET AUDIOSET DATA ####
		audioset_files = get_dir_files(path=audioset_path, just_filename=True, print_info=False)
		# Keep just the ones with extension .h5 and NOT unbalanced data!
		audioset_files = [fi for fi in audioset_files if (fi.split('.')[-1] == 'h5') and (fi.split('_')[0] != 'unbal')] #@@@@ modify if we want the strong from the unbalanced too

		# Get 'bal_tain' and 'eval' in the same dict
		audioset_data = read_h5_files(path=audioset_path, files=audioset_files, print_info=True)
		#### END - GET AUDIOSET DATA ####



		#### START - CREATE STRONG OUTPUT #### @@@@
		dcase_data
		#### END - CREATE STRONG OUTPUT ####



		#### START - REDUCE OUTPUT SIZE #### @@@@
		p = '../02456_project_audioset_attention/'
		project_files = get_dir_files(path=p, just_filename=True, print_info=False)
		
		# Keep just the ones with extension .csv
		project_files = [fi for fi in project_files if (fi.split('.')[-1] == 'csv')]

		a, new_output_classes = read_csv_files(path=p, files=[project_files[0]], delimiter=',', header=0, print_info=True, extract_video_id=False)
		print('\n\n----------------------------')
		print_dict_shape(dict_data=new_output_classes, title='Mapping')
		audioset_data = reduce_output_size(audioset_data, new_output_classes)
		#### END - REDUCE OUTPUT SIZE ####



		#### START - SELECTING STRONGLY LABELED DATA ####
		print('\n\n----------------------------')
		print_dict_shape(dict_data=dcase_str_lab_data, title='DCASE data shape')
		print_audioset_shape(audioset_data)
		print('\n\n----------------------------')
		dataset = concatenate_matches(dcase_str_lab_data, audioset_data)
		print('\nMatches: ', len(dataset['video_id']), '/', len(dcase_str_lab_data['video_id']))
		print_dict_shape(dict_data=dataset, title='dataset shape')
		#### END - SELECTING STRONGLY LABELED DATA ####


		#### START - CREATE STRONGLY OUTPUT (ys) ####

		# @@@@ CHANGE THIS FUNCTION FOR ADEX'S CODE!!!
		"""
		path = '../dcase2018_baseline/task4/dataset/metadata/test/'
		files = ['test.csv']
		test_idvs, test_out = create_strong_output(path, files) 

		path = '../dcase2018_baseline/task4/dataset/metadata/eval/'
		files = ['eval.csv']
		eval_idvs, eval_out = create_strong_output(path, files)

		# concatenate and store in dict
		strong_out = {}
		strong_out['ys'] = np.concatenate((test_out, eval_out),axis=0)
		strong_out['video_id'] = np.concatenate((test_idvs, eval_idvs),axis=0)
		"""
		#### END - CREATE STRONGLY OUTPUT (ys) ####


		# Last step: save in the 'h5' file

		# save train/test data
		for fn in audioset_data.keys():
			pn = 'data/'+fn
			fx = h5py.File(pn, 'a') # read/create the file
			print("\n\n  Save for training/testing - ",fn,':\n')
			save2h5(fx, audioset_data[fn])

		# save strong out
		"""
		file_strong_out = 'strong_out.h5'
		pn = 'data/'+file_strong_out
		fx = h5py.File(pn, 'a') # read/create the file
		print("\n\n  Save for training/testing - ",file_strong_out,':\n')
		save2h5(fx, strong_out)
		"""
		# general dataset (old)
		save2h5(f, dataset)
	
	#### END - SAVE IN 'H5' FILE ####
	return dataset

if __name__ == '__main__':

	dataset_file_name = 'file_name_to_save_data.h5'
	dcase_path = '../dcase2018_baseline/task4/dataset/metadata/'
	audioset_path = '../packed_features/'
	dataset = gen_data(dataset_file_name, dcase_path, audioset_path)
