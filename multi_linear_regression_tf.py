import numpy as np
import tensorflow as tf
# X1, X2, X3, y
data = np.array([
[ 73, 80, 75, 152],
[ 93, 88, 93, 185],
[ 89, 91, 90, 180],
[ 96, 98, 100, 196],
[ 73, 66, 70, 142]
], dtype=np.float32) / 200
x = data[:, :-1]
y = data[:, [-1]]
w = tf.Variable(tf.random_normal(shape=(3,1)))
b = tf.Variable(0.0)
h = tf.matmul(x, w) + b #(5,3)*(3,1) = (5,1)
cost = tf.reduce_mean((h - y)**2)
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#모든 변수를 초기화해라
    print('h', sess.run(h), 'cost', sess.run(cost))
    for i in range(100):
        sess.run(train)
    print('h', sess.run(h), 'cost', sess.run(cost))
    print('w b',sess.run(w),sess.run(b))



