from PIL import Image
import sys
import numpy as np
from scipy.misc import imsave, imresize
import cv2
from sklearn.linear_model import LogisticRegression
from math import sqrt
import pickle
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from random import randint, shuffle
from sklearn.cluster import KMeans
from multiprocessing import Process, Pipe
import math
from preprocessing import arrayToMatrix,\
	matrixToArray

TRAIN_RAW_IMAGE_1 = '/home/mircea/Desktop/Licenta/part2/segmentation/train/3126702.jpg'
TRAIN_RESULT_IMAGE_1 = '/home/mircea/Desktop/Licenta/part2/segmentation/train/3126702.bmp'

TRAIN_RAW_IMAGE_2 = '/home/mircea/Desktop/Licenta/part2/segmentation/train/3127062.jpg'
TRAIN_RESULT_IMAGE_2 = '/home/mircea/Desktop/Licenta/part2/segmentation/train/3127062.bmp'

TRAIN_RAW_IMAGE_3 = '/home/mircea/Desktop/Licenta/part2/segmentation/train/3127782.jpg'
TRAIN_RESULT_IMAGE_3 = '/home/mircea/Desktop/Licenta/part2/segmentation/train/res_3127782.bmp'

TEST_RAW_IMAGE_1 = '/home/mircea/Desktop/Licenta/part2/segmentation/test/3127362.jpg'
TEST_RESULT_IMAGE_1 = '/home/mircea/Desktop/Licenta/part2/segmentation/test/3127362.bmp'

TEST_RAW_IMAGE_2 = '/home/mircea/Desktop/Licenta/part2/segmentation/test/3135612.jpg'
TEST_RESULT_IMAGE_2 = '/home/mircea/Desktop/Licenta/part2/segmentation/test/3135612.bmp'

TEST_RAW_IMAGE_3 = '/home/mircea/Desktop/Licenta/part2/segmentation/test/3128352.jpg'
TEST_RESULT_IMAGE_3 = '/home/mircea/Desktop/Licenta/part2/segmentation/test/res_3128352.bmp'

LOGISTIC_REGRESSION = 0
SVM = 1
ANN = 2


def buildFeatureOnPart(conn, n, m, start_index, end_index, red_vec, green_vec,\
	blue_vec, x_feature_modif, y_feature_modif, additional_feat_n):

	if additional_feat_n != 0:
		additional_feats = conn.recv()

	# print(len(additional_feats[0]))
	# print(len(additional_feats[1]))

	inputs = []

	i = start_index / m
	j = start_index % m

	for ind in range(start_index, end_index):
		inputs.append([red_vec[ind], green_vec[ind], blue_vec[ind],])

		if additional_feat_n != 0:
			for add_feat in additional_feats:
				inputs[-1].append(add_feat[ind])

		for i_mod, j_mod in zip(x_feature_modif, y_feature_modif):
			feat_i, feat_j = int(i + i_mod), int(j + j_mod)

			if 0 <= feat_i < n and 0 <= feat_j < m:
				feat_ind = feat_i * m + feat_j

				inputs[-1] += [red_vec[feat_ind],\
					green_vec[feat_ind], blue_vec[feat_ind],]

				if additional_feat_n != 0:
					for add_feat in additional_feats:
						inputs[-1].append(add_feat[feat_ind])

			else:
				inputs[-1] += [0, 0, 0,]
				if additional_feat_n != 0:
					for add_feat in additional_feats:
						inputs[-1].append(0)

		if j + 1 < m:
			j += 1
		else:
			i += 1
			j = 0

	conn.send(inputs)
	conn.close()

def buildFeatureArray_2(red_mat, green_mat, blue_mat, radius_vector, granulation=4,\
	proc_no=8, additional_feats=[]):
	red_vec, green_vec, blue_vec =\
		matrixToArray(red_mat),\
		matrixToArray(green_mat),\
		matrixToArray(blue_mat)

	n = len(red_mat)
	m = len(red_mat[0])

	alpha = 0
	angular_step = math.pi / granulation
	angles = []

	for _ in range(granulation):
		angles.append(alpha)
		alpha += angular_step

	x_feature_modif = []
	y_feature_modif = []

	for radius in radius_vector:
		for alpha in angles:
			x_feature_modif.append(radius * math.cos(alpha))
			y_feature_modif.append(radius * math.sin(alpha))

	assigned_nums = [0 for _ in range(proc_no)]
	ex_no = n * m
	i = 0

	while ex_no != 0:
		assigned_nums[i] += 1
		ex_no -= 1
		i = (i + 1) % proc_no

	pipe_and_proc = []

	i = 0
	for an in assigned_nums:
		parent_conn, child_conn = Pipe()

		p = Process(target=buildFeatureOnPart,\
			args=(\
				child_conn,\
				n, m,\
				i, i + an,\
				red_vec, green_vec, blue_vec,\
				x_feature_modif, y_feature_modif,\
				len(additional_feats),\
			))
		p.start()

		pipe_and_proc.append((parent_conn, p,))

		if len(additional_feats) != 0:
			parent_conn.send(additional_feats)

		i += an

	inputs = []
	for parent_conn, p in pipe_and_proc:
		inputs += parent_conn.recv()

	for parent_conn, p in pipe_and_proc:
		p.join()

	return inputs

def buildTestImage(image, value=255):

	red_v = []

	green_v = []

	blue_v = []

	for i in range(len(image[0])):

		red_v.append([])

		green_v.append([])

		blue_v.append([])

		for j in range(len(image[0][0])):

			if image[0][i][j] > 160\
				and image[1][i][j] > 230\
				and image[2][i][j] > 140:

				red_v[i].append(value)

				green_v[i].append(value)

				blue_v[i].append(value)


			else:

				red_v[i].append(0)

				green_v[i].append(0)

				blue_v[i].append(0)

	return (red_v, green_v, blue_v)

def shuffleTheInputs(input_vec, output_vec):
	comb = list(zip(input_vec, output_vec))
	shuffle(comb)
	new_input_vec = []
	new_output_vec = []
	for e1, e2 in comb:
		new_input_vec.append(e1)
		new_output_vec.append(e2)

	return (new_input_vec, new_output_vec,)

def ballanceTheInputs(input_vec, output_vec, scale_factor=1):
	new_input_vec = []
	new_output_vec = []

	counter = 0
	for X, y in zip(input_vec, output_vec):
		if y == 1:
			counter += 1
			new_input_vec.append(X)
			new_output_vec.append(y)

	counter = int(scale_factor * counter)

	visited = []
	while counter != 0:

		n = randint(0, len(output_vec) - 1)

		while n in visited:
			n = randint(0, len(output_vec) - 1)

		visited.append(n)

		new_input_vec.append(input_vec[n])
		new_output_vec.append(output_vec[n])

		counter -= 1

	return shuffleTheInputs(new_input_vec, new_output_vec)

RADIUS_ARRAY = [1.75, 2.75, 3.75, 4.75, 5.75, 6.75,]
# RADIUS_ARRAY = [1.75, 3.75, 5.75, 7.75, 9.75, 11.75]
# RADIUS_ARRAY = [1.5, 7, 28, 56]
# RADIUS_ARRAY = [3.5, 7, 14, 28, 56, 112]
# RADIUS_ARRAY = [1.5, 3, 6, 12, 24, 48]
# RADIUS_ARRAY = [1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5, 25.5]

# input_tuple, output_tuple := tuple of matrixes
def get_model(input_tuple, output_tuple, model_type=LOGISTIC_REGRESSION,\
	percentage=float(1), additional_feats=[]):

	output_vec = []
	input_vec = []

	for m, n in zip(output_tuple, input_tuple):
		output_vec += matrixToArray(m[0], lambda el: 1 if el == 255 else 0)

		if len(n) == 3:
			input_vec += buildFeatureArray_2(n[0], n[1], n[2],\
				RADIUS_ARRAY,)
		else:
			input_vec += buildFeatureArray_2(n[0], n[1], n[2],\
				RADIUS_ARRAY, additional_feats=n[3:])

	ex_no = int(percentage * len(output_vec))

	input_vec, output_vec = shuffleTheInputs(input_vec,\
		output_vec)

	input_vec, output_vec = (input_vec[:ex_no],\
		output_vec[:ex_no],)

	# input_vec, output_vec = ballanceTheInputs(input_vec, output_vec)

	if model_type == LOGISTIC_REGRESSION:
		engine = LogisticRegression(solver='sag', max_iter=200,\
			tol=0.001)
	elif model_type == SVM:
		# engine = svm.SVC(kernel='linear', max_iter=200)
		# engine = svm.SVC(C=0.1, kernel='linear', max_iter=200)
		# engine = svm.SVC(C=10, kernel='linear', max_iter=200)
		# engine = svm.SVC(kernel='poly', max_iter=2000, degree=5)
		engine = svm.SVC(kernel='rbf', max_iter=2000,)
	else:
		# engine = MLPClassifier()
		# engine = MLPClassifier(hidden_layer_sizes=(50,50,))
		engine = MLPClassifier(hidden_layer_sizes=(50,50, 50, 50))


	print('Finished building feature array and will commence training...')

	engine.fit(input_vec, output_vec)

	print('Finished training !')

	return engine

def test(additional_feats=[]):

	ra_1 = readImage(TRAIN_RAW_IMAGE_1)
	re_1 = readImage(TRAIN_RESULT_IMAGE_1)

	ra_2 = readImage(TRAIN_RAW_IMAGE_2)
	re_2 = readImage(TRAIN_RESULT_IMAGE_2)

	# ra_3 = readImage(TRAIN_RAW_IMAGE_3)
	# re_3 = readImage(TRAIN_RESULT_IMAGE_3)

	# image = readImage(sys.argv[1])

	# ex_img = buildTestImage(image)

	# save_rgb_img(\
	#     np.array(ex_img[0]).transpose(),\
	#     np.array(ex_img[1]).transpose(),\
	#     np.array(ex_img[2]).transpose(),\
	#     'res_' + sys.argv[1][:-3] + 'bmp',\
	# )

	# save_rgb_img(\
	#     np.array(re_1[0]),\
	#     np.array(re_1[1]),\
	#     np.array(re_1[2]),\
	#     'plm.bmp',\
	# )

	engine = get_model((ra_1, ra_2,), (re_1, re_2,), model_type=ANN, percentage=0.99)
	# engine = get_model((ra_3,), (re_3,), model_type=LINEAR_REGRESSION, percentage=1)

	test_percentage = float(1)

	ra_1 = readImage(TEST_RAW_IMAGE_1)
	re_1 = readImage(TEST_RESULT_IMAGE_1)

	ra_2 = readImage(TEST_RAW_IMAGE_2)
	re_2 = readImage(TEST_RESULT_IMAGE_2)

	# ra_3 = readImage(TEST_RAW_IMAGE_3)
	# re_3 = readImage(TEST_RESULT_IMAGE_3)

	input_vec = []
	# input_vec += buildFeatureArray(ra_1[0], ra_1[1], ra_1[2])
	# input_vec += buildFeatureArray(ra_2[0], ra_2[1], ra_2[2])
	# input_vec += buildFeatureArray(ra_3[0], ra_3[1], ra_3[2])

	input_vec += buildFeatureArray_2(ra_1[0], ra_1[1], ra_1[2],\
		RADIUS_ARRAY)

	ex_no = int(test_percentage * len(input_vec))

	output_vec = []
	output_vec += matrixToArray(re_1[0], lambda el: 1 if el == 255 else 0)
	# output_vec += matrixToArray(re_2[0], lambda el: 1 if el == 255 else 0)
	# output_vec += matrixToArray(re_3[0], lambda el: 1 if el == 255 else 0)

	print('Will start predicting...')

	predicted_vec = engine.predict(input_vec[:ex_no])

	counter = float(0)
	for y, p in zip(output_vec[:ex_no], predicted_vec[:ex_no]):
		if y == p: counter += 1

	print('Accuracy: ' + str(counter/ex_no))
	# print('Counter: ' + str(counter))

	predicted_mat = arrayToMatrix( predicted_vec, len(re_1[0]), len(re_1[0][0]),\
		lambda el: 255 if el == 1 else 0)

	save_rgb_img(\
	    np.array(predicted_mat).transpose(),\
	    np.array(predicted_mat).transpose(),\
	    np.array(predicted_mat).transpose(),\
	    'pred.bmp',\
	)