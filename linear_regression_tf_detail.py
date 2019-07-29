import numpy as np
import tensorflow as tf
# y = w*x + b 선형방정식을 학습시키자
tf.enable_eager_execution()#session 없이 사용 가능
x = np.array([150, 160, 170, 180, 185])/200
y = np.array([50, 55, 60, 68, 72])/100
w = tf.Variable(0.4)
b = tf.Variable(0.0)
learning_rate = 0.01

for i in range(100):
    with tf.GradientTape() as tape:
        hypothesis = w * x + b  # 가설, 예측
        cost = tf.reduce_mean(tf.square(hypothesis - y))  # MSE

    w_grad, b_grad = tape.gradient(cost, [w,b])
    #w -= learning_rate * w_grad
    w.assign_sub(learning_rate * w_grad) #assign 값을 할당, 바꿔라
    b.assign_sub(learning_rate * b_grad)
    if i%10==0: print(i, w.numpy(), b.numpy(), cost)