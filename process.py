#plotly

import os, glob
import itertools
import logging
import shutil
import random
import csv
import uuid
import base64
from itertools import product
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import boto3

from image_processing import *
#from train_test import *

#botocore.errorfactory.ProvisionedThroughputExceededException: An error occurred (ProvisionedThroughputExceededException) when calling the CreateCollection operation: Provisioned rate exceeded

# CHANGE THIS SETTING TO BE COMPREHENSIVE (SLOW) OR SMALL (FAST)
RUN_TYPE = 'COMPREHENSIVE'

if RUN_TYPE=='COMPREHENSIVE':
	params = {'resolution': [240, 160, 80],#, 40],
	'bit_depth': list(range(3,8)),
	'number_faces': [100, 300, 600],
	'number_training_images': [1,2,3]
	}
	# 3x5x3x3 = 135 total combinations of parameters
	# test & training images should have same resolution & depth
	# Rekognition: The minimum pixel resolution for height and width is 80 pixels
	# http://docs.aws.amazon.com/rekognition/latest/dg/limits.html
	parameter_batches = [dict(zip(params.keys(), combo)) for combo in product(params['resolution'], params['bit_depth'], params['number_faces'], params['number_training_images'])]
	# but that takes forever to run (>6 hours), and could cost up to $500 on AWS ($.001/image)
	# so instead we'll just vary each parameter once, while keeping lower levels of other params
elif RUN_TYPE=='SMALL':
	parameter_batches = [
	# best possible parameters
	{'resolution': 240, 'bit_depth': 8, 'number_faces': 100, 'number_training_images': 3},
	# vary resolution, keep other params fixed at min
	{'resolution': 240, 'bit_depth': 3, 'number_faces': 100, 'number_training_images': 1},
	{'resolution': 160, 'bit_depth': 3, 'number_faces': 100, 'number_training_images': 1},
	{'resolution': 80, 'bit_depth': 3, 'number_faces': 100, 'number_training_images': 1},
	# vary bit depth, keep other params fixed at min
	{'resolution': 80, 'bit_depth': 4, 'number_faces': 100, 'number_training_images': 1},
	{'resolution': 80, 'bit_depth': 5, 'number_faces': 100, 'number_training_images': 1},
	{'resolution': 80, 'bit_depth': 6, 'number_faces': 100, 'number_training_images': 1},
	{'resolution': 80, 'bit_depth': 7, 'number_faces': 100, 'number_training_images': 1},
	{'resolution': 80, 'bit_depth': 8, 'number_faces': 100, 'number_training_images': 1},
	# vary number faces, keep other params fixed at min
	{'resolution': 80, 'bit_depth': 3, 'number_faces': 300, 'number_training_images': 1},
	{'resolution': 80, 'bit_depth': 3, 'number_faces': 600, 'number_training_images': 1},
	# vary number training images, keep other params fixed at min
	{'resolution': 80, 'bit_depth': 3, 'number_faces': 100, 'number_training_images': 2},
	{'resolution': 80, 'bit_depth': 3, 'number_faces': 100, 'number_training_images': 3}]

"""elif RUN_TYPE=='REMAINING':
	parameter_batches = [
	{'resolution': 240, 'bit_depth': 6, 'number_faces': 600, 'number_training_images': 1},
	{'resolution': 240, 'bit_depth': 6, 'number_faces': 300, 'number_training_images': 3},
	{'resolution': 240, 'bit_depth': 6, 'number_faces': 300, 'number_training_images': 2},
	{'resolution': 240, 'bit_depth': 6, 'number_faces': 300, 'number_training_images': 1},
	{'resolution': 240, 'bit_depth': 6, 'number_faces': 100, 'number_training_images': 3},
	{'resolution': 240, 'bit_depth': 6, 'number_faces': 100, 'number_training_images': 2},
	{'resolution': 240, 'bit_depth': 6, 'number_faces': 100, 'number_training_images': 1},
	{'resolution': 240, 'bit_depth': 5, 'number_faces': 600, 'number_training_images': 3},
	{'resolution': 240, 'bit_depth': 5, 'number_faces': 600, 'number_training_images': 2},
	{'resolution': 240, 'bit_depth': 4, 'number_faces': 300, 'number_training_images': 3}]"""

rekognition = boto3.client('rekognition', aws_access_key_id='REPLACE_ME',
	aws_secret_access_key='REPLACE_ME')

def select_faces():
	"""
	pick subset which have at least 4 images/face
	"""
	try:
		logger.debug('in select faces')
		#unique = set()
		#grouped_faces = defaultdict(list)
		selected_faces = []
		for dirName, subdirList, fileList in os.walk(os.path.join(os.getcwd(), 'input_images')):
		#for file in glob.glob('input_images/**/*.jpg', recursive=True):
			#logger.debug('Found directory: %s' % dirName)
			#for fname in fileList:
				#logger.debug('\t%s' % fname)
			if len(fileList)>=4:
				logger.debug('found face with >= 4 images')
			#	if not selected_faces.get(dirName):
			#logger.debug(file)
			#unique.add(file)
				selected_faces.append({dirName: fileList})
			#logger.debug(fileList)
		#logger.debug(unique)
		#files = list(unique)
		#logger.debug(len(files))

		# group by name
		#for file in files:
			#logger.debug(file)
		#	parts = file.split('/')
		#	group = parts[1]
		#	grouped_faces[group].append(file)
		# if >=4, select
		#for key, values in grouped_faces.items():
		#	if len(values)>=4:
				#logger.debug('found face with >= 4 images: ' + str(values))
		#		selected_faces[key] = values

		return selected_faces
	except Exception as e:
		logger.error(e, exc_info=True)

def run_sweep_parameters(selected_faces):
	"""
	test/training random split
	"""
	try:
		for batch in parameter_batches:
			# clean out directory
			#shutil.rmtree(os.path.join(os.getcwd(), 'current_images/*'))
			batch['uuid'] = str(uuid.uuid4())
			if not os.path.exists(os.path.join(os.getcwd(), 'current_images/'+ batch['uuid'])):
				os.makedirs(os.path.join(os.getcwd(), 'current_images/'+ batch['uuid']))
			files = glob.glob(os.path.join(os.getcwd(), 'current_images/'+ batch['uuid'] + '/*'))
			for f in files:
				os.remove(f)
			#faces = [{key: values} for key, values in selected_faces.items()]
			faces = selected_faces[:batch['number_faces']]
			process_images(batch, faces)
		
		with ThreadPoolExecutor(max_workers=25) as executor:
			results = executor.map(train_and_test, parameter_batches)
		all_results = {result['key']: result['values'] for result in results if result}
		return all_results
	except Exception as e:
		logger.error(e, exc_info=True)

def process_images(params, image_list):
	"""
	
	"""
	try:
		for image in image_list:
			for folder, files in image.items():
				for file in files:
					logger.debug(file)
					file_path = os.path.join(os.getcwd(), folder, file)
					#new_path = os.path.dirname(os.path.dirname(folder))
					new_path = os.path.join(os.getcwd(), 'current_images', params['uuid'], file)
					#logger.debug(file_path)
					logger.debug(new_path)
					convert(file_path, new_path, params['resolution'], params['bit_depth'])
	except Exception as e:
		logger.error(e, exc_info=True)

def train_and_test(params):
	"""
	create_collection
	index_faces
	search_faces_by_image

	create faceset
	search faceset by image
	key: T_jymdLVLnenAKYQUPGaNxvJscPfsPW_
	secret: lWckA6amTi96tymIDBYsMaxSfAPacJ_0
	"""
	try:
		key = 'round_9' + '_'.join(['resolution',str(params['resolution']),'bit_depth',str(params['bit_depth']),'number_faces',str(params['number_faces']),'number_training_images',str(params['number_training_images'])])
		#key = 'test12'
		logger.debug('collection: ' + key)
		response = rekognition.create_collection(CollectionId=key)
		logger.debug(response)

		all_images = defaultdict(list)

		number_training = params.get('number_training_images')
		# iterate through all images
		# group by name
		files = glob.glob(os.path.join(os.getcwd(), 'current_images/' + params['uuid'] + '/*'))
		for f in files:
			name_parts = f.split('_')
			name = name_parts[1].split('images/')[1] + '_' + name_parts[2]
			#logger.debug(name)
			all_images[name].append(f)

		# pick n for training set, where n = training
		# the rest go into test set
		# training = [{'name': ['image1', 'image2']}]
		# testing = [{'name': ['image3']}]
		training = defaultdict(list)
		testing = defaultdict(list)
		for item, values in all_images.items():
			random.shuffle(values)
			training[item] = values[:number_training]
			if RUN_TYPE=='SMALL':
				testing[item] = values[number_training:number_training+1] # to speed things up, we only test with a single image per face, not all of them
			else:
				#testing[item] = values[number_training:]
				testing[item] = values[number_training:number_training+1]

		# add training images
		for name, values in training.items():
			logger.debug(name)
			face_name = name.split('/')[1]
			for f in values:
				logger.debug('uploading training file: ' + f)
				with open(f, "rb") as image_file:
					#encoded_string = base64.b64encode(image_file.read())
					response = rekognition.index_faces(
						CollectionId=key,
						Image={
							'Bytes': image_file.read(),
						},
						ExternalImageId=face_name,#Tom_Hanks, etc.
						DetectionAttributes=['DEFAULT']
					)

					logger.debug(response)
					#{"FaceModelVersion":"2.0","FaceRecords":[{"Face":{"BoundingBox":{"Height":0.4615384638309479,"Left":0.26923078298568726,"Top":0.25641027092933655,"Width":0.4615384638309479},"Confidence":98.2965087890625,"ExternalImageId":"Debbie_Reynolds","FaceId":"83dac949-0f6c-4013-8a4a-3f2d61c8b140","ImageId":"ef7d79ea-7ee8-578a-98fc-7aecf0e4f161"},"FaceDetail":{"BoundingBox":{"Height":0.4615384638309479,"Left":0.26923078298568726,"Top":0.25641027092933655,"Width":0.4615384638309479},"Confidence":98.2965087890625,"Landmarks":[{"Type":"eyeLeft","X":0.416293740272522,"Y":0.426862508058548},{"Type":"eyeRight","X":0.5635715126991272,"Y":0.44325920939445496},{"Type":"nose","X":0.4649758040904999,"Y":0.5060997009277344},{"Type":"mouthLeft","X":0.4082030653953552,"Y":0.5807798504829407},{"Type":"mouthRight","X":0.5241758823394775,"Y":0.5921143889427185}],"Pose":{"Pitch":6.999231338500977,"Roll":6.720160007476807,"Yaw":-11.311697006225586},"Quality":{"Brightness":24.786460876464844,"Sharpness":99.99990844726562}}}],"OrientationCorrection":"ROTATE_0"}

		confidence_results = []
		# search with test images
		for name, values in testing.items():
			logger.debug(name)
			face_name = name.split('/')[1]
			for f in values:
				with open(f, "rb") as image_file:
					#encoded_string = base64.b64encode(image_file.read())

					try:
						response = rekognition.search_faces_by_image(
							CollectionId=key,
							Image={
								'Bytes': image_file.read(),
							},
							MaxFaces=1,
							FaceMatchThreshold=0
						)
						logger.debug(response)
					except rekognition.exceptions.InvalidParameterException as e:
						# botocore.errorfactory.InvalidParameterException: An error occurred (InvalidParameterException) when calling the SearchFacesByImage operation: There are no faces in the image. Should be at least 1.
						confidence_results.append({'actual': face_name, 'predicted': None, 'confidence': None, 'precision': None, 'recall': 0})
						logger.error(e, exc_info=True)
						break

					#keys: SearchedFaceBoundingBox, SearchedFaceConfidence, FaceMatches
					recall = False
					matches = response.get('FaceMatches')
					if matches:
						for match in matches:
							face = match.get('Face')
							if face:
								recall = True
								predicted_name = face.get('ExternalImageId')
								confidence = face.get('Confidence')
								confidence_results.append({'actual': face_name, 'predicted': predicted_name, 'confidence': confidence, 'precision': face_name==predicted_name, 'recall': 1})
					if not recall:
						confidence_results.append({'actual': face_name, 'predicted': None, 'confidence': None, 'precision': None, 'recall': 0})

		# return stats
		return {'key': '_'.join(['resolution',str(params['resolution']),'bit_depth',str(params['bit_depth']),'number_faces',str(params['number_faces']),'number_training_images',str(params['number_training_images'])]), 'values': confidence_results}
	except Exception as e:
		logger.error(e, exc_info=True)

def save_and_plot_results(results):
	"""
	10-fold cross validation
	accuracy vs. parameters
	ROC curves

	accuracy vs. training images
	accuracy vs. number faces
	accuracy vs. bit depth
	accuracy vs. resolution
	"""
	try:
		# save as CSV
		keys = ['actual','predicted','confidence','precision','recall']

		for key in results.keys():
			fname = 'results/' + key + '.csv'
			with open(fname, 'w') as csvfile:
				logger.debug('writing csv file: ' + fname)
				writer = csv.DictWriter(csvfile, fieldnames=keys)
				writer.writeheader()
				for values in results[key]:
					writer.writerow(values)
		
		# save a summary CSV too
		# sum up precision and recall to get %
		fname = 'results/summary.csv'
		with open(fname, 'w') as csvfile:
			logger.debug('writing csv file: ' + fname)
			writer = csv.DictWriter(csvfile, fieldnames=['resolution','bit_depth','number_faces','number_training_images','precision','recall'])
			writer.writeheader()
			for key, values in results.items():
				parts = key.split('_')
				total = len(values)
				precision = sum([value['precision'] for value in values if value['precision']!=None])
				recall = sum([value['recall'] for value in values])
				# TP/TP+FP
				precision_percent = float(precision/sum([1 for value in values if value['precision']!=None]))
				# TP/TP+FN
				recall_percent = float(recall/total)
				writer.writerow({'resolution': parts[1], 'bit_depth': parts[4], 'number_faces': parts[7], 'number_training_images': parts[11], 'precision': precision_percent, 'recall': recall_percent})

		# plot 4
	except Exception as e:
		logger.error(e, exc_info=True)

if __name__ == "__main__":
	selected_faces = select_faces()
	all_results = run_sweep_parameters(selected_faces)
	save_and_plot_results(all_results)