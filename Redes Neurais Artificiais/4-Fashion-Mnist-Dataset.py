import numpy as np
import datetime
import tensorflow as tf
from keras.datasets import fashion_mnist

(X_treino, y_treino), (X_teste, y_teste) = fashion_mnist.load_data()
print(X_treino[0])

X_treino = X_treino / 255.0
X_teste = X_teste / 255.0

print("aaa")
print(X_treino[0])