import scipy.io
import os
import numpy as np
import h5py
import skimage.io
import cv2

aa = skimage.io.imread('CAR196/car_ims/000001.jpg')
img_aa = cv2.resize(aa, (256, 256))

train_data_x = []
train_data_y = []
test_data_x = []
test_data_y = []

data = scipy.io.loadmat('CAR196/cars_annos.mat')
annotations = data['annotations']

for i in range(annotations.shape[1]):
    name = str(annotations[0,i][0])[2:-2]
    test  = int(annotations[0,i][6])
    label = int(annotations[0,i][5]) - 1
    filename = os.path.join('CAR196', name)
    I = skimage.io.imread(filename)
    I = cv2.resize(I, (256, 256))
    if len(I.shape) != 3:
        img2 = np.zeros_like(img_aa)
        img2[:, :, 0] = I
        img2[:, :, 1] = I
        img2[:, :, 2] = I
        I = img2
    if test == 0:
        train_data_x.append(I.tolist())
        train_data_y.append(label)
    else:
        test_data_x.append(I.tolist())
        test_data_y.append(label)

print(np.shape(train_data_x))
print(np.shape(train_data_y))
print(np.shape(test_data_x))
print(np.shape(test_data_y))

datafile = h5py.File('CAR196.h5', 'w')
datafile.create_dataset("train_data_pixel", dtype = 'uint8', data=train_data_x)
datafile.create_dataset("train_data_label", dtype = 'int64', data=train_data_y)
datafile.create_dataset("valid_data_pixel", dtype = 'uint8', data=test_data_x)
datafile.create_dataset("valid_data_label", dtype = 'int64', data=test_data_y)
datafile.close()
print("Save data finish!!!")