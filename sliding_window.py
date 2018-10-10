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
import os
from collections import deque
from preprocessing import readImage, matrixToArray, arrayToMatrix,\
	save_rgb_img
from shutil import rmtree

TRAIN_RAW_IMAGE_1 = './train_dotted/3126702.jpg'
TRAIN_RESULT_IMAGE_1 = './train_dotted/3126702.bmp'

TRAIN_RAW_IMAGE_2 = './train_dotted/3127062.jpg'
TRAIN_RESULT_IMAGE_2 = './train_dotted/3127062.bmp'

TRAIN_RAW_IMAGE_3 = './train_dotted/3127782.jpg'
TRAIN_RESULT_IMAGE_3 = './train_dotted/res_3127782.bmp'

TEST_RAW_IMAGE_1 = './test_dotted/3127362.jpg'
TEST_RESULT_IMAGE_1 = './test_dotted/3127362.bmp'

# TEST_RAW_IMAGE_2 = '/home/mircea/Desktop/Licenta/part2/segmentation/test/3135612.jpg'
# TEST_RESULT_IMAGE_2 = '/home/mircea/Desktop/Licenta/part2/segmentation/test/3135612.bmp'

TEST_RAW_IMAGE_3 = './test_dotted/3128352.jpg'
TEST_RESULT_IMAGE_3 = './test_dotted/res_3128352.bmp'

IMAGES_ARRAY = (\
	(TRAIN_RAW_IMAGE_1, TRAIN_RESULT_IMAGE_1),\
	(TRAIN_RAW_IMAGE_2, TRAIN_RESULT_IMAGE_2),\
	(TRAIN_RAW_IMAGE_3, TRAIN_RESULT_IMAGE_3),\
	(TEST_RAW_IMAGE_1,TEST_RESULT_IMAGE_1),\
	# (TEST_RAW_IMAGE_2,TEST_RESULT_IMAGE_2),\
	(TEST_RAW_IMAGE_3,TEST_RESULT_IMAGE_3),\
)

def cropImage(im, i_pos, j_pos, width, height, save_name='f.jpg'):

	n, m = len(im[0]), len(im[0][0])

	if width % 2 == 0 or height % 2 == 0:
		print('Pleas provide uneven dimensions !')
	else:
		red_v = []
		green_v = []
		blue_v = []

		for i in range(i_pos - int((width-1) / 2), i_pos + int((width-1) / 2) + 1):
			red_v.append([])
			green_v.append([])
			blue_v.append([])

			for j in range(j_pos - int((height-1) / 2), j_pos + int((height-1) / 2) + 1):
				if 0 <= i < n and 0 <= j < m:
					red_v[-1].append(im[0][i][j])
					green_v[-1].append(im[1][i][j])
					blue_v[-1].append(im[2][i][j])
				else:
					red_v[-1].append(0)
					green_v[-1].append(0)
					blue_v[-1].append(0)


		if save_name != 'f.jpg':
			save_rgb_img(\
			    np.array(red_v).transpose(),\
			    np.array(green_v).transpose(),\
			    np.array(blue_v).transpose(),\
			    save_name,\
			)

		return (red_v, green_v, blue_v)

def cropImageWithAdditFeat(im, i_pos, j_pos, width, height, save_name='f.jpg'):

	n, m = len(im[0]), len(im[0][0])

	if width % 2 == 0 or height % 2 == 0:
		print('Pleas provide uneven dimensions !')
	else:
		red_v = []
		green_v = []
		blue_v = []
		add_v = []

		for i in range(i_pos - int((width-1) / 2), i_pos + int((width-1) / 2) + 1):
			red_v.append([])
			green_v.append([])
			blue_v.append([])
			add_v.append([])

			for j in range(j_pos - int((height-1) / 2), j_pos + int((height-1) / 2) + 1):
				if 0 <= i < n and 0 <= j < m:
					red_v[-1].append(im[0][i][j])
					green_v[-1].append(im[1][i][j])
					blue_v[-1].append(im[2][i][j])
					add_v[-1].append(im[3][i][j])
				else:
					red_v[-1].append(0)
					green_v[-1].append(0)
					blue_v[-1].append(0)
					add_v[-1].append(0)

		return (red_v, green_v, blue_v, add_v,)

def getCellCenters(image, c_i):
	coordModifiers = ((-1, 0), (0, -1), (0, 1), (1, 0),)

	if c_i == 0:
		non_c_i = (1, 2,)
	elif c_i == 1:
		non_c_i = (0, 2,)
	else:
		non_c_i = (0, 1,)

	n = len(image[0])
	m = len(image[0][0])

	cell_centers_vec = []

	outside_pixels = []
	for i in range(n):
		outside_pixels.append([])
		for j in range(m):
			outside_pixels[-1].append(True)

	for i in range(n):
		for j in range(m):
			if image[c_i][i][j] == 255\
				and image[non_c_i[0]][i][j] == 0\
				and image[non_c_i[1]][i][j] == 0:

				i_sum = j_sum = counter = 0

				outside_pixels[i][j] = False
				queue = deque([(i, j)])
				visited = [(i, j,)]
				while len(queue) != 0:
					coords = queue.popleft()

					for coordModifier in coordModifiers:
						newI = coords[0] + coordModifier[0]
						if 0 <= newI < n:
							newJ = coords[1] + coordModifier[1]
							if 0 <= newJ < m and\
								not (image[0][newI][newJ] == 255\
								and image[1][newI][newJ] == 255\
								and image[2][newI][newJ] == 255)\
								and (newI, newJ,) not in visited:
								queue.append((newI, newJ))
								visited.append((newI, newJ,))

								if outside_pixels[newI][newJ]:
									outside_pixels[newI][newJ] = False

									for k in range(1,10):

										if newI - k >= 0 and outside_pixels[newI - k][newJ]:
											outside_pixels[newI - k][newJ] = False

										if newI + k < n and outside_pixels[newI + k][newJ]:
											outside_pixels[newI + k][newJ] = False

										if newJ - k >= 0 and outside_pixels[newI][newJ - k]:
											outside_pixels[newI][newJ - k] = False

										if newJ + k < m and outside_pixels[newI][newJ + k]:
											outside_pixels[newI][newJ + k] = False

								counter += 1
								i_sum += newI
								j_sum += newJ

				cell_centers_vec.append((\
					int(float(i_sum)/counter),\
					int(float((j_sum)/counter)),\
				))

	ou = []
	for i in range(n):
		for j in range(m):
			if outside_pixels[i][j]:
				ou.append((i, j,))

	outside_centers_vec = []
	for _ in range(len(cell_centers_vec)):
		outside_centers_vec.append(ou[randint(0, len(ou))])

	return zip(cell_centers_vec, outside_centers_vec,)

def getSlidingWindowExamples(dim):

	examples_dir = './sliding_ex/'

	if os.path.isdir(examples_dir):
		rmtree(examples_dir)

	os.makedirs(examples_dir)


	for raw_im_name, res_im_name in IMAGES_ARRAY:
		dir_name = raw_im_name[-11:-4]

		os.mkdir(examples_dir + dir_name)

		raw_im = readImage(raw_im_name)
		res_im = readImage(res_im_name)

		# print(res_im)

		a = 0

		for cen_1, cen_0 in getCellCenters(res_im, 2):

			cropImage(raw_im,\
				cen_1[0], cen_1[1], dim, dim,\
				examples_dir + dir_name + '/1_' + str(a) + '.bmp')

			cropImage(raw_im,\
				cen_0[0], cen_0[1], dim, dim,\
				examples_dir + dir_name + '/0_' + str(a) + '.bmp')

			a += 1

def getSlidingWindow(img, dim, centers):
	rez = []

	for cen_1 in centers:

		rez.append(cropImage(img,\
			cen_1[0], cen_1[1], dim, dim,))

	return rez

def procJob1(conn, n, m, an, red_vec, green_vec,\
	blue_vec, sq_dim,):

	engine = conn.recv()

	additional_features = conn.recv()
		#vector de matrice

	red_mat, green_mat, blue_mat =\
		arrayToMatrix(red_vec, n, m,),\
		arrayToMatrix(green_vec, n, m,),\
		arrayToMatrix(blue_vec, n, m,)

	inputs = []
	for i in range(0, len(an), 2):

		inputs.append([])

		for ii in range(an[i], an[i] + sq_dim):
			for jj in range(an[i+1], an[i+1] + sq_dim):
				if 0 <= ii < n and 0 <= jj < m:
					inputs[-1] += [red_mat[ii][jj], green_mat[ii][jj],\
						blue_mat[ii][jj]]

					for add_feat in additional_features:
						inputs[-1].append(add_feat[ii][jj])
				else:
					inputs[-1] += [0, 0, 0,]

					for add_feat in additional_features:
						inputs[-1].append(0)

	kkt = engine.predict_proba(inputs)

	if isinstance(kkt[0], np.ndarray):
		kkt = list(map(lambda el: el[1], kkt))

	conn.send(kkt)

def multipleSquares(sq_dim, n, m):
	if sq_dim % 2 == 0:
		print("Even sq_dim !")
		return []

	pozitions = []
	hf_sq_dim = int((sq_dim-1)/2)

	for i in range(-hf_sq_dim, n, sq_dim):
		for j in range(-hf_sq_dim, m, sq_dim):
			pozitions += [\
				i, j,\
				i + hf_sq_dim, j,\
				i, j + hf_sq_dim,\
			]

	return pozitions

def test_sliding(sq_dim,\
	p_inputs, n_inputs, test_inputs):

	poz_inputs = []
	neg_inputs = []
	# test_inputs = []

	# for p_inp, n_inp, t_inp in zip(p_inputs, n_inputs, t_inputs):
	for p_inp, n_inp in zip(p_inputs, n_inputs):
		inp = []
		for r, g, b, p in zip(\
				matrixToArray(p_inp[0]),\
				matrixToArray(p_inp[1]),\
				matrixToArray(p_inp[2]),\
				matrixToArray(p_inp[3]),\
			):
			inp += [r, g, b, p,]
		poz_inputs.append(inp)

		inp = []
		for r, g, b, p in zip(\
				matrixToArray(n_inp[0]),\
				matrixToArray(n_inp[1]),\
				matrixToArray(n_inp[2]),\
				matrixToArray(n_inp[3]),\
			):
			inp += [r, g, b, p,]
		neg_inputs.append(inp)

	model_type = 2

	if model_type == 0:
		engine = LogisticRegression()
	elif model_type == 1:
		engine = svm.SVC()
	else:
		engine = MLPClassifier()

	train_proc = 1
	proc_no = 7

	train_num = int(train_proc * len(poz_inputs))

	shuffle(neg_inputs)
	shuffle(poz_inputs)

	inputs = []
	outputs = []

	for p, n in zip(poz_inputs, neg_inputs):

		inputs += [p, n,]
		outputs += [1, 0,]

		if train_num == 0:
			break

		train_num -= 1

	engine.fit(inputs, outputs)

	print('Finished fiting !')

	red_mat, green_mat, blue_mat = test_inputs[:3]

	red_vec, green_vec, blue_vec =\
		matrixToArray(red_mat),\
		matrixToArray(green_mat),\
		matrixToArray(blue_mat)

	n = len(red_mat)
	m = len(red_mat[0])

	frames_coords = multipleSquares(sq_dim, n, m)

	assigned_nums = [[] for _ in range(proc_no)]
	i = 0

	for j in range(0, len(frames_coords), 2):
		assigned_nums[i] += [frames_coords[j],\
			frames_coords[j+1],]
		i = (i + 1) % proc_no

	pipe_and_proc = []

	for an in assigned_nums:
		parent_conn, child_conn = Pipe()

		p = Process(target=procJob1,\
			args=(\
				child_conn,\
				n, m,\
				an,\
				red_vec, green_vec, blue_vec,\
				sq_dim,\
			))
		p.start()

		parent_conn.send(engine)

		parent_conn.send([] if len(test_inputs) == 3 else\
			test_inputs[3:])

		pipe_and_proc.append((parent_conn, p,))

	pred_probs = []

	for parent_conn, p in pipe_and_proc:

		kkt = parent_conn.recv()

		pred_probs += kkt

	for parent_conn, p in pipe_and_proc:
		p.join()

	windows = list(zip([i for i in range(len(pred_probs))], pred_probs))

	windows.sort(reverse=True, key=lambda el: el[1])

	a = 0

	for ind, prob in windows:
		i = int(ind / m)
		j = ind % m
		if a % 5 == 0:
			cropImage(\
				(red_mat, green_mat, blue_mat,),\
				i + int((sq_dim-1)/2),\
				j + int((sq_dim-1)/2),\
				sq_dim, sq_dim, './rez/' + str(a) + '.bmp'\
			)

		a += 1