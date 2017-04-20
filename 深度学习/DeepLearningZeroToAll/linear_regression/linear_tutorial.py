import numpy as np
from keras.layers import Dense
from keras.models import Sequential

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

model = Sequential()
model.add(Dense(1, input_dim=1))
model.compile(loss='mse', optimizer='sgd')

model.summary()

model.fit(x_train, y_train, epochs=1000)
y_predict = model.predict(np.array([2]))
print(y_predict)
