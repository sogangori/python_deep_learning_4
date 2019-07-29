import numpy as np
import keras#pip install keras
m = 5
x = np.array([150, 160, 170, 180, 185])/200
y = np.array([50, 55, 60, 68, 72])
키_test = np.array([165, 190])/200
# y = w*x + b 선형방정식을 학습시키자
model = keras.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))#키 1개로 몸무게 1개 예측
model.compile(loss='mse', optimizer='sgd')
model.fit(x, y, epochs=100)
print(model.predict(키_test))


