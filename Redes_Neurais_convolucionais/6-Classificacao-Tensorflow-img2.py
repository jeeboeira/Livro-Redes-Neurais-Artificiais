import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

nomes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

(X_treino, y_treino), (X_teste, y_teste) = cifar10.load_data()

X_treino = X_treino / 255.0
X_teste  = X_teste  / 255.0

plt.imshow(X_teste[55])

modelo = tf.keras.models.Sequential()

modelo.add(tf.keras.layers.Conv2D(filters     = 32,
                                  kernel_size = 3,         # Cada um para um canal de cor
                                  padding     = 'same',    # Estipula técnica de preenchimento onde houverem campos faltando informação
                                  activation  = 'relu',
                                  input_shape = [32, 32, 3]))
modelo.add(tf.keras.layers.Conv2D(filters     = 32,
                                  kernel_size = 3,
                                  padding     = 'same',
                                  activation  = 'relu'))
modelo.add(tf.keras.layers.MaxPool2D(pool_size = 2,        # Tamanho em px de cada bloco a ser mapeado
                                     strides   = 2,        # Quantas 'casas' o interpretador deverá separar, ler e mapear cada bloco de px.
                                     padding   = 'valid')) # Só utiliza dados de blocos íntegros, regiões de borda ou onde não se mantém consistência são desconsiderados.
# Expande os filtros de 32 para 64
modelo.add(tf.keras.layers.Conv2D(filters     = 64,
                                  kernel_size = 3,
                                  padding     = 'same',
                                  activation  = 'relu'))
modelo.add(tf.keras.layers.Conv2D(filters     = 64,
                                  kernel_size = 3,
                                  padding     = 'same',
                                  activation  = 'relu'))
modelo.add(tf.keras.layers.MaxPool2D(pool_size = 2,        # Tamanho em px de cada bloco a ser mapeado
                                     strides   = 2,        # Quantas 'casas' o interpretador deverá separar, ler e mapear cada bloco de px.
                                     padding   = 'valid')) # Só utiliza dados de blocos íntegros, regiões de borda ou onde não se mantém consistência são desconsiderados.
modelo.add(tf.keras.layers.Flatten())

#===============#
# Compile & Fit #
#===============#

modelo.add(tf.keras.layers.Dense(units      = 128,                            
                                 activation = 'relu'))
modelo.add(tf.keras.layers.Dense(units      = 10,                            
                                 activation = 'softmax'))
modelo.compile(loss      = 'sparse_categorical_crossentropy', # Função de erro/perda em como os resultados se dispersam para cada uma das 10 probabilidades.
               optimizer = 'Adam', 
               metrics   = ['sparse_categorical_accuracy'])   # Categorização baseado na melhor margem de precisão possível

print(modelo.summary())

modelo.fit(X_treino,
           y_treino,
           epochs              = 50,
           steps_per_epoch     = 1000,
           use_multiprocessing = True)

