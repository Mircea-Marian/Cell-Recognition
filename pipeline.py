from preprocessing import *
from som_segmentation import som_segmentation
from PIL import Image
from segm import test, buildFeatureArray_2, get_model,\
	RADIUS_ARRAY
from sliding_window import getCellCenters, IMAGES_ARRAY,\
	cropImageWithAdditFeat, test_sliding
import pickle

# Training and test images aliases
TRAIN_RAW_IMAGE_1 = './train/3126702.jpg'
TRAIN_RESULT_IMAGE_1 = './train/3126702.bmp'

TRAIN_RAW_IMAGE_2 = './train/3127062.jpg'
TRAIN_RESULT_IMAGE_2 = './train/3127062.bmp'

TRAIN_RAW_IMAGE_3 = './train/3127782.jpg'
TRAIN_RESULT_IMAGE_3 = './train/res_3127782.bmp'

TEST_RAW_IMAGE_1 = './test/3127362.jpg'
TEST_RESULT_IMAGE_1 = './test/3127362.bmp'

TEST_RAW_IMAGE_2 = './test/3135612.jpg'
TEST_RESULT_IMAGE_2 = './test/3135612.bmp'

TEST_RAW_IMAGE_3 = './test/3128352.jpg'
TEST_RESULT_IMAGE_3 = './test/res_3128352.bmp'

def test_stage_0():
	"""Represents the function used for testing the performance of the
	preprocessing algorithms aswell as the
	"""
	ra_1 = readImage(TRAIN_RAW_IMAGE_1)
	re_1 = readImage(TRAIN_RESULT_IMAGE_1)

	ra_2 = readImage(TRAIN_RAW_IMAGE_2)
	re_2 = readImage(TRAIN_RESULT_IMAGE_2)

	# Uncomment below if more examples are required.
	# ra_3 = readImage(TRAIN_RAW_IMAGE_3)
	# re_3 = readImage(TRAIN_RESULT_IMAGE_3)

	# Uncomment below if the additional features are needed.
	# ra_1 += (
	# 	laplace_operator(TRAIN_RAW_IMAGE_1),\
	# 	# k_means(TRAIN_RAW_IMAGE_1)[0],\
	# 	)

	# Uncomment below if the additional features are needed.
	# ra_2 += (
	# 	laplace_operator(TRAIN_RAW_IMAGE_2),\
	# 	# k_means(TRAIN_RAW_IMAGE_2)[0],\
	# 	)

	# The prediction model is obtained and trained.
	engine = get_model((ra_1, ra_2,), (re_1, re_2,), model_type=SVM, percentage=0.1)

	test_percentage = float(1) # how many tests

	ra_1 = readImage(TEST_RAW_IMAGE_1)

	# Uncomment below if the additional features are needed.
	# ra_1 += (
	# 	laplace_operator(TEST_RAW_IMAGE_1),\
	# 	# k_means(TEST_RAW_IMAGE_1)[0],\
	# 	)

	re_1 = readImage(TEST_RESULT_IMAGE_1)

	# ra_2 = readImage(TEST_RAW_IMAGE_2)
	# re_2 = readImage(TEST_RESULT_IMAGE_2)

	input_vec = []
	# The features are extracted.
	input_vec += buildFeatureArray_2(ra_1[0], ra_1[1], ra_1[2],\
		RADIUS_ARRAY,\
		additional_feats=([] if len(ra_1) == 3 else ra_1[3:]))

	ex_no = int(test_percentage * len(input_vec)) # actual number of the test sample

	output_vec = []
	output_vec += matrixToArray(re_1[0], lambda el: 1 if el == 255 else 0)

	print('Will start predicting...')

	predicted_vec = engine.predict(input_vec[:ex_no])

	counter = float(0)
	for y, p in zip(output_vec[:ex_no], predicted_vec[:ex_no]):
		if y == p: counter += 1

	print('Accuracy: ' + str(counter/ex_no))

	predicted_mat = arrayToMatrix( predicted_vec, len(re_1[0]), len(re_1[0][0]),\
		lambda el: 255 if el == 1 else 0)

	# The predicted segmentation is saved.
	save_rgb_img(\
	    np.array(predicted_mat).transpose(),\
	    np.array(predicted_mat).transpose(),\
	    np.array(predicted_mat).transpose(),\
	    'pred.bmp',\
	)

def test_pipeline(sq_dim):

	# Images are read into matrixes.
	ra_1 = readImage(TRAIN_RAW_IMAGE_1)
	re_1 = readImage(TRAIN_RESULT_IMAGE_1)

	ra_2 = readImage(TRAIN_RAW_IMAGE_2)
	re_2 = readImage(TRAIN_RESULT_IMAGE_2)

	# List vectors are added.
	ra_1 += (\
		laplace_operator(TRAIN_RAW_IMAGE_1),\
		k_means(TRAIN_RAW_IMAGE_1)[0],)

	ra_2 += (\
		laplace_operator(TRAIN_RAW_IMAGE_2),\
		k_means(TRAIN_RAW_IMAGE_2)[0],)

	print('Finished first stage ...')

	engine = get_model((ra_1, ra_2,), (re_1, re_2,), model_type=LOGISTIC_REGRESSION, percentage=0.25)

	print('Got model ...')

	t_ra_1 = readImage(TEST_RAW_IMAGE_1)
	t_ra_1 += (\
		laplace_operator(TEST_RAW_IMAGE_1),\
		k_means(TEST_RAW_IMAGE_1)[0],)

	input_vec = []

	input_vec += buildFeatureArray_2(t_ra_1[0], t_ra_1[1], t_ra_1[2],\
		RADIUS_ARRAY,\
		additional_feats=([] if len(t_ra_1) == 3 else t_ra_1[3:]))

	predicted_mat = arrayToMatrix( engine.predict(input_vec), len(re_1[0]), len(re_1[0][0]),\
		lambda el: 255 if el == 1 else 127)

	print('Finished second stage ...')

	poz_inputs = []
	neg_inputs = []

	ra_1 = ra_1[:3] + (\
		list(map(\
			lambda row: list(map(lambda el: 255 if el == 255 else 127, row)),\
			re_1[0]\
		)),\
	)
	zipped_centers_1 = getCellCenters(\
		readImage(IMAGES_ARRAY[0][1]), 2)
	for cen_1, cen_0 in zipped_centers_1:
		poz_inputs.append(cropImageWithAdditFeat(ra_1,\
			cen_1[0], cen_1[1], sq_dim, sq_dim))

		neg_inputs.append(cropImageWithAdditFeat(ra_1,\
			cen_0[0], cen_0[1], sq_dim, sq_dim))



	ra_2 = ra_2[:3] + (\
		list(map(\
			lambda row: list(map(lambda el: 255 if el == 255 else 127, row)),\
			re_2[0]\
		)),\
	)
	zipped_centers_2 = getCellCenters(\
		readImage(IMAGES_ARRAY[1][1]), 2)
	for cen_1, cen_0 in zipped_centers_2:
		poz_inputs.append(cropImageWithAdditFeat(ra_2,\
			cen_1[0], cen_1[1], sq_dim, sq_dim))

		neg_inputs.append(cropImageWithAdditFeat(ra_2,\
			cen_0[0], cen_0[1], sq_dim, sq_dim))


	t_ra_1 = t_ra_1[:3] + (predicted_mat,)

	test_sliding(sq_dim, poz_inputs, neg_inputs, t_ra_1)

	print('Finished third stage ...')

if __name__ == "__main__":
	test_pipeline(55)


