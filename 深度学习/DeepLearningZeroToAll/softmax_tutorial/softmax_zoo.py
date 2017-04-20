import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils

xy = np.loadtxt('zoo.csv', delimiter=',')
x_data = xy[:, 0:-1]
y_data = xy[:, -1]
nb_classes = 7
y_one_hot = np_utils.to_categorical(y_data, nb_classes)

model = Sequential()
model.add(Dense(7, input_shape=(16,), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(x_data, y_one_hot, epochs=10000)
pred = model.predict_classes(x_data)
# for p, y in zip(pred, y_data):
#     print(p, '>>>>>>>>>>', y)
acc = history.history.get('acc')
loss = history.history.get('loss')
from matplotlib import pyplot as plt

plt.plot(range(len(loss)), loss, color='red')
plt.plot(range(len(acc)), acc)
plt.show()
