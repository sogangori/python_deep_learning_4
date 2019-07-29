import numpy as np
import tensorflow as tf
#파일 업로드 sogangori/python_deep_learning_4
m = 5
x = np.array([150, 160, 170, 180, 185])/200
y = np.array([50, 55, 60, 68, 72])/100
# y = w*x + b 선형방정식을 학습시키자
w = tf.Variable(0.4) #초기값 1.0
b = tf.Variable(0.0)
h = w * x + b
cost = (1/m) * tf.reduce_sum((h - y)**2)
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())#모든 변수를 초기화해라
print('h', sess.run(h), 'cost', sess.run(cost))
for i in range(100):
    sess.run(train)#동작 수행
print('h', sess.run(h), 'cost', sess.run(cost))
print('w b',sess.run(w),sess.run(b))


