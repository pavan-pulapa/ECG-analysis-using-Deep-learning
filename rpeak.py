import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
import scipy.io 
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

batch_size = 500
num_classes = 2
epochs = 12

seed = 7

from scipy.io import loadmat
X = scipy.io.loadmat('X_train.mat')
I=X['input_train']
X2 = scipy.io.loadmat('Y_train.mat')
Y=X2['output_train']
X3=scipy.io.loadmat('X_test.mat')
I_t=X3['I']

X3=scipy.io.loadmat('Y_test.mat')
Y_t=X3['O']
X22= scipy.io.loadmat('X_test1.mat')
print X22
X_test1=X22['input_test']
x_train = I.reshape(27435, 241,1).astype('float32')
x_test = I_t.reshape(4000, 241,1).astype('float32')
x_test1= X_test1.reshape(10000, 241,1).astype('float32')
y_train=Y
y_test=Y_t
model = Sequential()
model.add(Conv1D(64, kernel_size=10,
                 activation='relu',
                 input_shape=(241,1)))

model.add(MaxPooling1D(pool_size=4))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)


y_pred=model.predict(x_test1)
plt.plot(y_pred)

plt.show()
print('Test loss:', score[0])
print('Test accuracy:', score[1])
