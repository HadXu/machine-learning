import numpy as np
from keras.layers import Dense
from keras.models import Sequential

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd')
model.fit(x_data, y_data, epochs=1000)
print("1,2", model.predict_classes(np.array([[2, 1]])))
