import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('./data/driving_log.csv') as f:
	reader = csv.reader(f)
	for line in reader:
		lines.append(line)

images = []
measurements = []

for line in lines:
	if 'center' == line[0]:
		continue
	source_path = line[0]
	filename = source_path.split('\\')[-1]
	current_path = './data/IMG/' + filename
	
	image = cv2.imread(current_path)
	images.append(image)

	measurement = float(line[3])
	measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape = (160,320,3)))
model.add(Cropping2D(cropping = ((70, 25), (0, 0))))
model.add(Convolution2D(24,5,5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, nb_epoch = 5, validation_split = 0.2, shuffle = True)

model.save('model.h5')