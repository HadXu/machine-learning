import numpy as np
from keras.layers import Dense
from keras.models import Sequential

xy = np.loadtxt('data_01.csv', delimiter=',')
x_data = xy[:, 0:-1]
y_data = xy[:, -1]
model = Sequential()
model.add(Dense(1, input_dim=3))
model.compile(loss='mse', optimizer='rmsprop')
model.fit(x_data, y_data, epochs=10000)
print("0, 2, 1", model.predict(np.array([[0, 2, 1]])))
print("0, 9, -1", model.predict(np.array([[0, 9, -1]])))
