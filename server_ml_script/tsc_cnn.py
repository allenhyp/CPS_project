import numpy as np
from skimage import io, color, exposure, transform
import os
import glob
import h5py
from keras.models import Sequential, model_from_json, load_model
from keras.utils import np_utils
from keras import backend as K
import time
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
K.set_image_data_format("channels_last")
K.tensorflow_backend._get_available_gpus()
IMG_SIZE = 32

def preprocess_img(img):
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

def main():
	file = open('runtime.txt','a')
	model = load_model("traffic_sign_classifier_model.h5")
	for j in range(5):
		main_start_time = time.time()
		root_dir = '/home/allenhsu/uploads/'
		# root_dir = 'testing40/'
		X_test = []
		all_img_paths = glob.glob(os.path.join(root_dir, '*.png'))
		for img_path in all_img_paths:
			try:
				img = preprocess_img(io.imread(img_path))
				X_test.append(img)
			except(IOError, OSError):
				print('missed', img_path)
				pass
		X_test = np.array(X_test)
		file.write('preprocess_time: {0}\n'.format(time.time() - main_start_time))
		for i in range(10):
			classification_start_time = time.time()
			y_pred = model.predict_classes(X_test)
			# print("classification time: {}".format(time.time() - classification_start_time))
			# print("python script running time: {}".format(time.time() - main_start_time))
			# result = '{0}_{1}_{2}\n'.format(y_pred, time.time() - classification_start_time, time.time() - main_start_time)
			# print(result)
			model_runtime = time.time() - classification_start_time
			runtime = '{0}, {1} per image\n'.format(model_runtime, model_runtime/len(y_pred))

			file.write(runtime)
	file.close()

if __name__ == "__main__":
	main()