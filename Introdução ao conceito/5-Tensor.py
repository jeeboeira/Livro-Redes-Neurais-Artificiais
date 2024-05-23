import numpy as np
import tensorflow as tf

#Defino uma constante tensor
tensor = tf.constant([[32, 9], [5, 71]])
#Crio um array numpy
tensorn = np.array([[23, 4],[32, 51]])
#Transformo um array numpy em tensor constante
tensor_numpy = tf.constant(tensorn)
#Defino uma variavel tensor
tf_variavel = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
#Tensor com String
tf_string = tf.constant("JeSse")
tf_string2 = tf.constant(["Jessé", "python", "TensorFlow"])

print(tensor)
print(tensorn)
print(tensor_numpy)
print(tf_variavel)

#Manipulando variáveis
tf_variavel[0, 2].assign(100)
print(tf_variavel)

print (tensor + 2)
print (tensor * 2)
print (tensor + tensor)
print (tensor * tensor)
print (tensor / tensor)

print(tensor > tensorn)

print(np.square(tensor))
print(np.sqrt(tensor))
print(np.dot(tensor, tensor))
print(tf_string)

print(tf.strings.length(tf_string))

print(tf.strings.unicode_decode(tf_string, "UTF8"))

for i in tf_string2:
    print(i)