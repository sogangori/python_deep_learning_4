#순서 1. 데이터획득 2. 데이터 정제 3.데이터 분석 4. 특징 선택 5. 수치로 변환 6.정규화(scale)
#학습용/검증용 > 학습 > 성능
import pandas as pd #panel data 라이브러리
import numpy as np
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv('C:/Users/1/Downloads/winequality-red.csv', delimiter=';')#구분자
print('data', data.shape)
#print('data', data.iloc[0])
data = np.array(data)
x = data[:, :-1]
y = data[:, [-1]]
scaler = MinMaxScaler().fit(x)#값을 [0,1] 로 정규화
x = scaler.transform(x)#값을 [0,1] 로 정규화
x_original = scaler.inverse_transform(x)
print(np.min(x), np.max(x))
model = LinearRegression().fit(x, y)
score = model.score(x,y) #결정계수 R^2 1이 목표, 값이 작을 수록 예측이 나쁘다
print('score', score)
import tensorflow as tf
import tensorflow.contrib.slim as slim
x = tf.cast(x, tf.float32)
y = tf.cast(y, tf.float32)
h = slim.fully_connected(x, 1) #weight 가 자동으로 생성됩니다
cost = tf.reduce_mean((h - y)**2)
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#모든 변수를 초기화해라
    print('h', sess.run(h), 'cost', sess.run(cost))
    for i in range(100):
        sess.run(train)
    print('h', sess.run(h), 'cost', sess.run(cost))

