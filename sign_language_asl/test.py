#load model from *.h5
#test with batches from test sets
#plot confusion matrix and accuracy

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from tensorflow.keras.models import load_model
import itertools
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



test_path = '../organised/test/'

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(28,28), batch_size=10, shuffle=False)








new_model = load_model('sign_language.h5')
#print(new_model.summary())

predictions = new_model.predict(x=test_batches, steps=len(test_batches), verbose=0)


cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
plt.imshow(cm, interpolation='nearest')
for i in range(len(cm)):
    for j in range(len(cm)):
        text = plt.text(j, i, cm[i, j], ha="center", va="center", color="w")


cm1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#accuracy per class
print(cm1.diagonal())
sum = 0
#overall accuracy
for i in range(24):
    sum = sum + cm[i][i]
#accuray
print(sum/2400)
plt.tight_layout()
plt.show()

#print(cm)

