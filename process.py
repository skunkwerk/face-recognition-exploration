#plotly

import os
import itertools
import logging
from itertools import product

from image_processing import *
from train_test import *

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

params = {'resolution': [240, 174, 107, 40],
'bit_depth': list(range(3,8)),
'number_faces': [100, 300, 600],
'number_training_images': [1,2,3]
}
# 4x5x3x3 = 180 total combinations of parameters
# test & training images should have same resolution & depth

def select_faces():
	"""
	pick subset which have at least 2 images/face
	"""
	try:
		selected_faces = []
		for dirName, subdirList, fileList in os.walk(os.path.join(os.getcwd(), 'input_images')):
			#logger.debug('Found directory: %s' % dirName)
			#for fname in fileList:
				#logger.debug('\t%s' % fname)
			if len(fileList)>=4:
				logger.debug('found face with >= 4 images')
				selected_faces.append({dirName: fileList})
		logger.debug(len(selected_faces))
		return selected_faces
	except Exception as e:
		logger.error(e, exc_info=True)

def run_sweep_parameters():
	"""
	test/training random split
	"""
	try:
		parameter_groups = [dict(zip(params.keys(), combo)) for combo in product(params['resolution'], params['bit_depth'], params['number_faces'], params['number_training_images'])]
		for group in parameter_groups:

	except Exception as e:
		logger.error(e, exc_info=True)

def test_train_random_split():
	try:
		pass
	except Exception as e:
		logger.error(e, exc_info=True)

def plot_results():
	"""
	10-fold cross validation
	accuracy vs. parameters
	ROC curves
	"""
	try:
		pass
	except Exception as e:
		logger.error(e, exc_info=True)

if __name__ == "__main__":
	select_faces()
	run_sweep_parameters()