import csv
import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			correction = (0., .11, -.11)
			for batch_sample in batch_samples:
				for i in range(3):
					name = './data/IMG/'+batch_sample[i].split('\\')[-1]
					image = cv2.imread(name)
					center_angle = float(batch_sample[3])
					images.append(image)
					angles.append(center_angle + correction[i])
					images.append(np.fliplr(image))
					angles.append(-center_angle - correction[i])

			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

# read csv
lines = []
with open('./data/driving_log.csv') as f:
	reader = csv.reader(f)
	for line in reader:
		if 'center' == line[0]:
			continue
		lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

# using the Nvidia network with some dropout
model = Sequential()
# normalisation
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape = (160,320,3)))
# crop off top and bottom
model.add(Cropping2D(cropping = ((70, 25), (0, 0))))
model.add(Convolution2D(24,5,5, subsample = (2,2), activation = 'relu'))
model.add(Dropout(.25))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation = 'relu'))
model.add(Dropout(.25))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Dropout(.25))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Dropout(.25))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(.25))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')