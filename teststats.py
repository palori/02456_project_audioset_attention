import os
import gzip
import h5py
import time
import logging
from scipy import stats
from sklearn import metrics
import numpy as np



def calculate_stats(output, target, threshold):
	"""Calculate statistics including mAP, AUC, etc.

	Args:
	  output: 3d array, (samples_num, time, classes_num)
	  target: 3d array, (samples_num, time, classes_num)

	Returns:
	  stats: list of statistic of each class.
	"""

	classes_num = target.shape[0]
	timestep_num = target.shape[1]
	print(classes_num)
	print(timestep_num)
	stats = []

	# Class-wise statistics
	for j, k in [(j,k) for j in range(timestep_num) for k in range(classes_num)]:

		# Piecewise comparison
		output_rounded = output>threshold

		# Average precision
		avg_precision = metrics.average_precision_score(
			target[:, j, k], output_rounded[:, j, k], average=None)

		# AUC
		#auc = metrics.roc_auc_score(target[:, j, k], output_rounded[:, j, k], average=None)

		# Precisions, recalls
		(precisions, recalls, thresholds) = metrics.precision_recall_curve(
			target[:, j, k], output_rounded[:, j, k])

		# FPR, TPR
		(fpr, tpr, thresholds) = metrics.roc_curve(target[:, j, k], output_rounded[:, j, k])

		save_every_steps = 1000     # Sample statistics to reduce size
		dict = {'precisions': precisions[0::save_every_steps],
				'recalls': recalls[0::save_every_steps],
				'AP': avg_precision,
				'fpr': fpr[0::save_every_steps],
				'fnr': 1. - tpr[0::save_every_steps]}
				#'auc': auc}
		stats.append(dict)

	return stats

def test_calculate_stats():
	# a and b should have both 0 and 1 values for every batch
	a = np.array([[[1,1],[1,0]],[[0,1],[0,1]]])
	b = np.array([[[1,0],[1,0]],[[1,1],[0,1]]])
	threshold = 0.5
	stats = calculate_stats(a,b,threshold)
	print(stats)

if __name__ == '__main__':
	test_calculate_stats()