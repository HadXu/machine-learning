import numpy as np
from keras.layers import Dense
from keras.models import Sequential, optimizers

xy = np.loadtxt('diabets.csv', delimiter=',')
x_data = xy[:, 0:-1]
y_data = xy[:, -1]

model = Sequential()
model.add(Dense(1, input_dim=8, activation='sigmoid'))
opem = optimizers.SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=opem)
model.fit(x_data, y_data, epochs=2000)
print('predict............')
print(y_data[0], '>', model.predict_classes(np.array([x_data[0]])))
print(y_data[1], '>', model.predict_classes(np.array([x_data[1]])))
print(y_data[2], '>', model.predict_classes(np.array([x_data[2]])))
