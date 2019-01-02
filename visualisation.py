import os
import gzip
import h5py
import time
import logging
from scipy import stats
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import itertools

def barchart(stats, target, figure_name, fig=None, ax1=None, ax2=None, label='change', init=False, freq=False, time_ax=False):
	"""Random chance to be included inside 

	time_ax = True  -> time in X-axes
	        = False -> classes in X-axes
	"""
	print(stats)
	if time_ax:
		classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
	else:
		classes = ('Speech', 'Dog', 'Cat', 'Alarm', 'Dishes', 'Frying', 'Blender', 'Running\nwater', 'Vacuum\ncleaner', 'Tooth-\nbrush')
	print('classes:',classes)

	y_pos = np.arange(len(classes))
	auc = stats['auc']
	ap = stats['AP']

	frequency = np.array([np.sum(target[:,c,:]) for c in y_pos])
	#old_frequency = frequency
	frequency[::-1].sort()
	
	indexes = np.array([4,5,2,3,1,6,7,1,8,9])

	a0 = np.mean(auc, axis=0)
	a1 = np.mean(auc, axis=1)
	a0 = np.array([a0[i] for i in indexes])
	a1 = np.array([a1[i] for i in indexes])

	if time_ax:
		a = a0
		a0 = a1
		a1 = a

	ap0 = np.mean(ap, axis=0)
	ap1 = np.mean(ap, axis=1)
	ap0 = np.array([ap0[i] for i in indexes])
	ap1 = np.array([ap1[i] for i in indexes])
	
	if init :
		# Create a figure
		fig = plt.figure(1, clear=True, figsize=(8,6))
		ax1 = fig.subplots()
		ax2 = ax1.twinx()
	
		if freq:
			ax1.bar(y_pos, frequency, align='center', alpha=0.5, color='lightsteelblue')#'paleturquoise')
			#plt.yticks(y_pos, classes)
			ax1.set_ylabel('Frequency', color='b')
			ax1.tick_params('y', colors='b')
			#ax1.set_xlabel('Classes')
			tick_marks = np.arange(len(classes))
			plt.xticks(tick_marks, classes, rotation=45)
		else:
			plt.xticks(y_pos, classes)
			ax1.set_xlabel('Time [seconds]')

	# Roc
	#ax2.plot(ap0, '--r*')
	#ax2.plot(ap1, '--b*')
	#ax2.plot(a0, '--r*')
	if label == 'Single attention':
		ax2.plot(a1, '--*', color='darkblue', label = 'Single attention')
		ax2.plot(0.5*np.ones(10), '--g*', label = 'Random')
		#ax2.legend(loc=1, bbox_to_anchor=(0.6,0.5))
	elif label == 'Max pooling':
		ax2.plot(a0, '--*', color='darkorange', label = 'Max pooling')
		#ax2.legend(loc=1, bbox_to_anchor=(0.6,0.5))
	else:
		ax2.plot(a0, '--*', color='red', label = 'Avg pooling')
	ax2.set_ylabel('Area Under Curve', color='darkred')
	ax2.tick_params('y', colors='red')
	#ax2.set_xlabel('Classes')
	ax2.legend(loc = "bottom left")

	# Random line
	#if init:
		#ax2.legend(loc=1, bbox_to_anchor=(0.6,0.5))

	fig.show()
	fig.savefig(figure_name)
	return fig, ax1, ax2

def confusion_plot(output, target, confusionmode):
	"""Random chance to be included inside 
	"""

	if confusionmode == 'class':
		outputline = [np.ndarray.flatten(np.diag(np.arange(10))*output[k,:,:],'C') for k in range(output.shape[0])]
		targetline = [np.ndarray.flatten(np.diag(np.arange(10))*target[k,:,:],'C') for k in range(output.shape[0])]
		outputline = [item for sublist in outputline for item in sublist]
		targetline = [item for sublist in targetline for item in sublist]
		cm = metrics.confusion_matrix(targetline, outputline)
	elif confusionmode == 'time':
		outputline = [np.ndarray.flatten(output[k,:,:]*np.diag(np.arange(10)),'F') for k in range(output.shape[0])]
		targetline = [np.ndarray.flatten(target[k,:,:]*np.diag(np.arange(10)),'F') for k in range(output.shape[0])]
		outputline = [item for sublist in outputline for item in sublist]
		targetline = [item for sublist in targetline for item in sublist]
		cm = metrics.confusion_matrix(targetline, outputline)
		cm = cm[1::,1::]
	else : print('Error: problem with confusionmode argument')
	
	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	plt.figure()
	print('Confusion matrix')

	print(cm)

	plt.imshow(cm, interpolation='nearest')
	plt.title('Confusion matrix')
	plt.colorbar()
	classes = ('Speech', 'Dog', 'Cat', 'Alarm/bell/ringing', 'Dishes', 'Frying', 'Blender', 'Running water', 'Vacuum cleaner', 'Electric shaver/toothbrush')
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
			horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()



def calculate_stats(output, target, threshold):
	"""Calculate statistics including mAP, AUC, etc.

	Args:
	  output: 3d array, (samples_num, time, classes_num)
	  target: 3d array, (samples_num, time, classes_num)

	Returns:
	  stats: list of statistic of each class.
	"""

	classes_num = target.shape[2]
	timestep_num = target.shape[1]

	stats = {#'precisions': np.empty((10,10)),
			#	'recalls': np.empty((10,10)),
				'AP': np.zeros((10,10)),
			#	'fpr': np.empty((10,10)),
			#	'fnr': np.empty((10,10)),
				'auc': np.zeros((10,10))}

	#target = target.transpose(0,2,1)
	# Class-wise statistics
	for j, k in [(j,k) for j in range(timestep_num) for k in range(classes_num)]:
		#print((j,k))
		# Piecewise comparison
		#print('output: ', output)
		output_rounded = (output>threshold)*1
		#print('output_rounded: ', output_rounded)
		

		# Average precision
		avg_precision = metrics.average_precision_score(target[:, j, k], output_rounded[:, j, k], average=None)

		# AUC
		auc = metrics.roc_auc_score(target[:, j, k], output_rounded[:, j, k], average=None)

		# Precisions, recalls
		(precisions, recalls, thresholds) = metrics.precision_recall_curve(
			target[:, j, k], output_rounded[:, j, k])

		# FPR, TPR
		(fpr, tpr, thresholds) = metrics.roc_curve(target[:, j, k], output_rounded[:, j, k])

		save_every_steps = 1000     # Sample statistics to reduce size

		stats['AP'][j,k] = avg_precision
		stats['auc'][j,k] = auc
		

		"""
		dict = {'precisions': precisions[0::save_every_steps],
				'recalls': recalls[0::save_every_steps],
				'AP': avg_precision,
				'fpr': fpr[0::save_every_steps],
				'fnr': 1. - tpr[0::save_every_steps],
				'auc': auc}
		stats.append(dict)
		"""


	return stats

def test_calculate_stats():
	# a and b should have both 0 and 1 values for every batch
	a = np.array([[[1,0,0],[1,0,1]],[[0,1,0],[0,1,1]],[[0,1,1],[0,1,1]],[[0,1,0],[0,1,1]]])
	b = np.array([[[1,0,1],[1,0,1]],[[1,1,0],[0,1,1]],[[0,1,0],[0,1,1]],[[0,1,0],[0,1,1]]])
	threshold = 0.5
	stats = calculate_stats(a,b,threshold)
	print(stats)

def test_confusion_plot():
	a = np.ones((50,10,10))
	b = np.ones((50,10,10))
	confusionmode = 'time'
	confusion_plot(a,b,confusionmode)

def test_barchart(stats):
	barchart(stats)


def load_data(file2test):

	if file2test == 0:
		hdf5_path = 'data/eval1.h5'
		with h5py.File(hdf5_path, 'r') as hf:
			x = np.array(hf.get('x'))
			y = np.array(hf.get('y'))
		return x, y

	elif file2test == 1:
		hdf5_path = 'data/single_att.h5'
		with h5py.File(hdf5_path, 'r') as hf:
			output = np.array(hf.get('output'))
			cla = np.array(hf.get('cla'))
			norm_att = np.array(hf.get('norm_att'))
			mult = np.array(hf.get('mult'))
		return output, cla, norm_att, mult

	elif file2test == 2:
		hdf5_path = 'data/multy.h5'
		with h5py.File(hdf5_path, 'r') as hf:
			output = np.array(hf.get('output'))
			cla = np.array(hf.get('cla'))
			norm_att = np.array(hf.get('norm_att'))
			mult = np.array(hf.get('mult'))
			cla2 = np.array(hf.get('cla2'))
			norm_att2 = np.array(hf.get('norm_att2'))
			mult2 = np.array(hf.get('mult2'))
		return output, cla, norm_att, mult, cla2, norm_att2, mult2

	elif file2test >= 3:

		if file2test == 3:
			hdf5_path = 'data/pooling_average.h5'
		elif file2test == 4:
			hdf5_path = 'data/maxpooling.h5'
		with h5py.File(hdf5_path, 'r') as hf:
			output = np.array(hf.get('output'))
			b2 = np.array(hf.get('b2'))

			print('  4: output: ',output.shape)
			print('  4: b2: ',b2.shape)
		return output, b2


def normalization(data):
	print('data11: ', data)
	for k in data.keys():
		"""
		mini = np.ndarray.min(data[k])
		maxi = np.ndarray.max(data[k])
		data[k] = data[k]/(maxi-mini)
		print('mini: ',mini)
		print('maxi: ',maxi)
		"""
		data[k] = np.sqrt(np.sqrt(np.sqrt(data[k])))
		
	print('data22: ', data)
	return data

def test_real_data(time_ax=False, figure_name='foo2.png'):
	'''
	file2test:
	  0 ground truth
	  1 single attention
	  2 multi attention
	  3 average pooling
	  4 max pooling
	'''
	th = 0.8
	x, y = load_data(0)
	
	fig=None
	ax1=None
	ax2=None
	'''
	print('\n\n-------------------\n1')
	output, cla, norm_att, mult = load_data(1)
	y = y.transpose(0,2,1)
	data = {'target': y,
			'output': cla}
	#data = normalization(data)
	stats = calculate_stats(data['output'], data['target'], th)
	data['target'] = data['target'].transpose(0,2,1)
	
	fig, ax1, ax2 = barchart(stats, data['target'], figure_name='foo2.png', fig, ax1, ax2, label='Single attention', init=True, freq=False, time_ax=time_ax)
	
	
	print('\n\n-------------------\n2')
	output, cla, norm_att, mult, cla2, norm_att2, mult2 = load_data(2)
	cla = (cla+cla2)/2
	print(cla.shape)
	data = {'target': y,
			'output': cla}
	#data = normalization(data)
	stats = calculate_stats(data['output'], data['target'], th)
	barchart(stats, y)
	'''
	
	print('\n\n-------------------\n3')
	output, b2 = load_data(3)
	data = {'target': y,
			'output': b2}
	#data = normalization(data)
	stats = calculate_stats(data['output'], data['target'], th)
	if time_ax:
		fig, ax1, ax2 = barchart(stats, data['target'], figure_name, fig, ax1, ax2, 'Average pooling', init=True, freq=False, time_ax=time_ax)
	else:
		fig, ax1, ax2 = barchart(stats, data['target'], figure_name, fig, ax1, ax2, 'Average pooling', init=True, freq=True, time_ax=time_ax)
	
	
	print('\n\n-------------------\n4')
	output, b2 = load_data(4)
	data = {'target': y,
			'output': b2}
	#data = normalization(data)
	stats = calculate_stats(data['output'], data['target'], th)
	fig, ax1, ax2 = barchart(stats, data['target'], figure_name, fig, ax1, ax2, 'Max pooling', init=False, freq=False, time_ax=time_ax)
	
	print('\n\n-------------------\n1')
	output, cla, norm_att, mult = load_data(1)
	y = y.transpose(0,2,1)
	data = {'target': y,
			'output': cla}
	#data = normalization(data)
	stats = calculate_stats(data['output'], data['target'], th)
	data['target'] = data['target'].transpose(0,2,1)
	
	fig, ax1, ax2 = barchart(stats, data['target'], figure_name, fig, ax1, ax2, label='Single attention', init=False, freq=False, time_ax=time_ax)


if __name__ == '__main__':
	#stats = test_calculate_stats()
	#test_barchart(stats)

	test_real_data(time_ax=False, figure_name='figures/auc_classes.png')
	test_real_data(time_ax=True, figure_name='figures/auc_time.png')