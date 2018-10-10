import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imsave
from PIL import Image
from collections import deque
from sklearn.cluster import KMeans

LOGISTIC_REGRESSION = 0
SVM = 1
ANN = 2

def arrayToMatrix(vec, n, m, f=lambda x: x):
	imgRez = []
	k = 0
	for i in range(n):
		imgRez.append([])
		for j in range(m):
			imgRez[i].append(f(vec[k]))
			k = k + 1
	return imgRez

def matrixToArray(matrix, f=lambda x: x):
	rez = []
	for line in matrix:
		for el in line:
			rez.append(f(el))
	return rez

def save_rgb_img(red_v, green_v, blue_v, fname):
	imsave(fname, np.dstack((red_v, green_v, blue_v)))

def readImage(name=None):
	if name == None:
		return None
	im = Image.open(name).convert('RGB')
	m, n = im.size

	red_v = []

	green_v = []

	blue_v = []

	for i in range(m):

		red_v.append([])

		green_v.append([])

		blue_v.append([])

		for j in range(n):
			r, g, b = im.getpixel((i, j))

			red_v[i].append(r)

			green_v[i].append(g)

			blue_v[i].append(b)

	return (red_v, green_v, blue_v,)

def laplace_operator(src_filename, kernel_size=5):
	img = cv2.imread(src_filename,cv2.IMREAD_GRAYSCALE)

	filter = cv2.Laplacian(img,cv2.CV_64F, ksize=kernel_size)

	tr = np.transpose(filter)
	rez = []
	for line in tr:
		for el in line:
			rez.append(el)

	return rez

def clusteringTest(new_img, width, height):
	visited = [False for _ in new_img]

	d = {}
	for red, green, blue in new_img:
		if (red, green, blue,) not in d:
			d[ (red, green, blue,) ] = 1
		else:
			d[ (red, green, blue,) ] += 1

	d_list = list(d.items())
	d_list.sort(key=lambda el: el[1],reverse=True)

	print('Started surface counting !: ' +\
		str(width ) + ' ' + str(height))

	limit = 15

	surface_counter = 0
	bad_surface_few_pixels = 0
	bad_surface_punctured = 0
	for i in range(width):
		for j in range(height):
			index = i * height + j
			if not visited[index]:
				if new_img[index] == d_list[0][0]:
				# if False:
					visited[index] = True
				else:
					# coordModifiers = ((-1, -1), (-1, 0), (-1, 1), (0, -1),\
					# 	(0, 1), (1, -1), (1, 0), (1, 1))

					coordModifiers = ((-1, 0), (0, -1), (0, 1), (1, 0),)

					cluster_contour = []

					queue = deque([(i, j)])
					visited[index] = True

					pixel_counter = 1

					while len(queue) != 0:
						coords = queue.popleft()

						for coordModifier in coordModifiers:
							newI = coords[0] + coordModifier[0]
							if 0 <= newI < width:
								newJ = coords[1] + coordModifier[1]
								if 0 <= newJ < height\
									and not visited[newI * height + newJ]\
									and new_img[newI * height + newJ] == new_img[index]:

									pixel_counter += 1
									visited[newI * height + newJ] = True
									queue.append((newI, newJ))

									is_on_contour = False

									for cm2 in coordModifiers:
										new_newI = newI + cm2[0]
										new_newJ = newJ + cm2[1]

										if new_newI < 0 or new_newI == width\
											or new_newJ < 0 or new_newJ == height\
											or visited[new_newI * height + new_newJ]\
											or new_img[new_newI * height + new_newJ]\
											!= new_img[index]:
											is_on_contour = True
											break

									if is_on_contour:
										cluster_contour.append((newI, newJ))

					max_dist_sq = -1
					point1, point2 = (-1,-1,), (-1, -1,)

					for ii in range(len(cluster_contour)-1):
						for jj in range(ii + 1, len(cluster_contour)):
							new_dist_sq = (cluster_contour[ii][0] - cluster_contour[jj][0])**2\
								+ (cluster_contour[ii][1] - cluster_contour[jj][1])**2
							if new_dist_sq > max_dist_sq:
								max_dist_sq = new_dist_sq
								point1 = cluster_contour[ii]
								point2 = cluster_contour[jj]

					middle_point = (int( (float(point1[0]) + point2[0])/2 ),\
						int( (float(point1[1]) + point2[1])/2 ),)

					coordModifiers = ((-1, -1), (-1, 0), (-1, 1), (0, -1),\
						(0, 1), (1, -1), (1, 0), (1, 1))

					outer_pixels_counter = 0
					for ii, jj in cluster_contour:
						x_point, y_point = ii, jj

						while (x_point, y_point) != middle_point:
							if new_img[x_point * height + y_point] != new_img[index]:
								outer_pixels_counter += 1

							if outer_pixels_counter >= limit: break

							new_newI = x_point + coordModifiers[0][0]
							new_newJ = y_point + coordModifiers[0][1]
							min_dist_sq = (new_newI - middle_point[0])**2\
								+ (new_newJ - middle_point[1])**2

							for coordModifier in coordModifiers[1:]:
								newI = x_point + coordModifier[0]
								newJ = y_point + coordModifier[1]
								new_dist_sq = (newI - middle_point[0])**2\
									+ (newJ - middle_point[1])**2
								if new_dist_sq < min_dist_sq:
									min_dist_sq = new_dist_sq
									new_newI, new_newJ = newI, newJ

							x_point, y_point = new_newI, new_newJ

						if outer_pixels_counter >= limit: break

					# print(pixel_counter)

					if outer_pixels_counter >= limit:
						bad_surface_punctured += 1

					aaa = 50

					if pixel_counter > aaa:
						bad_surface_few_pixels += 1

					# if not has_encountered_outer_pixel and pixel_counter > 100:
					if outer_pixels_counter < limit and pixel_counter > aaa:
					# if outer_pixels_counter < limit:
						surface_counter += 1

					# surface_counter += 1

	print('bad_surface_punctured: ' + str(bad_surface_punctured))
	print('bad_surface_few_pixels: ' + str(bad_surface_few_pixels))
	print(surface_counter)

def k_means(src_filename):
	im = Image.open(src_filename)

	data = im.getdata()

	kmeans = KMeans(n_clusters=4, random_state=0, init='k-means++').fit(np.array(data))

	# new_img = []
	# for label in kmeans.labels_:
	# 	if label == 0:
	# 		new_img.append((0,0,0,))
	# 	elif label == 1:
	# 		new_img.append((255,0,0,))
	# 	elif label == 2:
	# 		new_img.append((0,255,0,))
	# 	else:
	# 		new_img.append((0,0,255,))

	new_img = []
	for label in kmeans.labels_:
		if label == 0:
			new_img.append(10)
		elif label == 1:
			new_img.append(63)
		elif label == 2:
			new_img.append(126)
		else:
			new_img.append(255)

	return (new_img, im.size[0], im.size[1],)