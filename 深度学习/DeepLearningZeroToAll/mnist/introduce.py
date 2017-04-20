import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils

np.random.seed(777)  # for reproducibility

from keras.datasets import mnist

nb_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], 784)
x_train = x_train.astype('float32') / 255


# one_hot
y_train = np_utils.to_categorical(y_train, nb_classes)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
x_test = x_test.astype('float32') / 255
# one_hot
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(nb_classes, input_dim=784, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=15)
score = model.evaluate(x_test, y_test)
print('\nAccuracy:', score[1])
print(history.history.get('acc'))
