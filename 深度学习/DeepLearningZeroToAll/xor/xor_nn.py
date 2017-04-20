from keras.layers import Dense
from keras.models import Sequential

x_data = [[0., 0.],
          [0., 1.],
          [1., 0.],
          [1., 1.]]
y_data = [[0.],
          [1.],
          [1.],
          [0.]]

model = Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
hist = model.fit(x_data, y_data, epochs=50000)
from matplotlib import pyplot as plt

acc = hist.history.get('acc')
plt.plot(range(len(acc)), acc)
plt.show()