from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np

x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5],
          [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

model = Sequential()
model.add(Dense(3,input_dim=3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.1),metrics=['accuracy'])
model.fit(x_data,y_data,epochs=200)
score = model.evaluate(x_test,y_test)
print(model.predict_classes(x_test))
print(np.argmax(model.predict(x_test),axis=0))
print(score[1])




