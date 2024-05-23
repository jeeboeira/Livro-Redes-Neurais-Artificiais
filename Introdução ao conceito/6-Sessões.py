import tensorflow.compat.v1 as tf # type: ignore
tf.disable_v2_behavior()

num1 = tf.constant(8)
num2 = tf.constant(15)

soma_num = num1 + num2

print(soma_num)
print(type(soma_num))

with tf.Session() as sess:
    soma = sess.run(soma_num)

print(soma)