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

# Aqui eu remodelo as dimensões, no caso aqui, havia uma matriz com 60000
    #amostras 28x28, e ficou uma base bidimensional
    #com 600000 amostras e 784 pontos de características referentes
    #a composição de cada imagem
print(X_treino.shape)
X_treino = X_treino.reshape(-1, 28*28)
print(X_treino.shape)
X_teste = X_teste.reshape(-1, 28*28)


modelo = tf.keras.models.Sequential()

print(modelo)

modelo.add(tf.keras.layers.Dense(units = 128, #Definido 128 neurônios
                                 activation = 'relu',
                                 input_shape = (784, ))) #Formato da camada de entrada no 
                                                            #formato dos dados normalizados
modelo.add(tf.keras.layers.Dropout(0.2)) # Estipula um número de amostras para serem eliminadas
                                            #entre camadas forçando a rede neural trabalhar
                                            #compensando a falta de dados. 0,2 - 20% das amostras
                                            #serão removidas desta camada para próxima.
modelo.add(tf.keras.layers.Dense(units = 10, # Denifino unidades de saída
                                 activation = 'softmax')) # O resultado do processamento gera uma
                                                            #probabilidade entre 10 categorias definidas.
modelo.compile(optimizer = 'adam', # Sempre ao fim de um rede neural, é necessário compilar ela.
               loss = 'sparse_categorical_crossentropy', # O processamento considera como métrica
                                                            #a proximidade/semelhança
               metrics = ['sparse_categorical_accuracy'])

#Resumo da estrutura
print(modelo.summary())


# Define númuero de treinos
modelo.fit(X_treino,
            y_treino,
            epochs = 100)

#Salva as configurações da rede neural, tanto estrutura quanto dados de aprendizado
modelo_json = modelo.to_json() # usada função to_json para salvar os parâmetros
with open("fashion_modelo_json", "w") as json_file: #pela função open() definimos o nome do arquivo, 'w' para ser lido sem incompatibilidades
    json_file.write(modelo_json) # Atualiza os conteúdo do arquivo

modelo.save_weights("fashion_modelo.h5") # Salva os peso, só aplicar o método save_weights, com extensão .h5
                                            #para leitura posterior, é só instanciar os dados por meio de uma variável
                                            #e chamar pela função lo.load_weights()
