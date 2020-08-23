#create and train model 
#study batch normalization 
#accuracy of 98%
#will save model as well 

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)





train_path = './organised/train'
valid_path = './organised/valid'
test_path = './orgainsed/test'


#create batches to preprocess data  / create batches /normalise image size(224,224) set classes
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(28,28), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(28,28),  batch_size=10)



model = Sequential()

model.add(Conv2D(64,(3,3),activation="relu",input_shape=(28,28,3)))
model.add(BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(2,2))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64,activation="relu"))

model.add(Dense(units=26, activation='softmax'))

print(model.summary())
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


#fit model for the data generated
model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=9,
    verbose=2
)
model.save("./sign_language.h5")
