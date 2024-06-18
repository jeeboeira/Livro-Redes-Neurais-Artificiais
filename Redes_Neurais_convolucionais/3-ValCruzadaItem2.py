import numpy as np
from keras.datasets import mnist
# Modelo pré configurado que cria e faz as conexões entre camadas de neurônios
from keras.models import Sequential
#Dense Cria as camadas da rede
#Conv2D cria camadas adaptadas ao processamento de imagens
#MaxPooling2D cria micromatrizes indexadas, para identificação
    #de padrões pixel a pixel
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
#Modelo de validação cruzada
from sklearn.model_selection import StratifiedKFold

seed = 5
np.random.seed(seed)

(etreino, streino),(eteste, steste) = mnist.load_data()

#FASE DE POLIMENTO

entradas = etreino.reshape(etreino.shape[0],28,28,1)
entradas = entradas.astype('float32')
entradas /= 255
saidas = np_utils.to_categorical(streino, 10)

#FIM DA FASE DE POLIMENTO

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
resultados = []

a = np.zeros(5)
b = np.zeros(shape = (saidas.shape[0], 1))

for evalcruzada, svalcruzada in kfold.split(entradas,
                                            np.zeros(shape = (saidas.shape[0],1))):
    classificador = Sequential()
    classificador.add(Conv2D(32,
                             (3,3),
                             input_shape = (28,28,1),
                             activation = 'relu'))
    classificador.add(MaxPooling2D(pool_size = (2,2)))
    #Transforma as matrizes para uma dimensão, para cada valor da linha atuar como um neurônio
    classificador.add(Flatten())
    classificador.add(Dense(units = 128,
                            activation = 'relu'))
    classificador.add(Dense(units = 10, # Camada de saida com 10 neurônios, já que a classificação
                                            #é de 0 a 9.
                            activation = 'softmax'))#Parecido com sigmoid, porém é capaz
                                            #de categorizar amostras em múltiplas saídas
    # COMPILADOR
    classificador.compile(loss = 'categorical_crossentropy', # Verifica quais dados não foram possíveis
                          #classificar quanto sua proximidade aos vizinhos.
                          optimizer = 'adam',
                          metrics = ['accuracy'])
    classificador.fit(entradas[evalcruzada], # Alimenta a rede
                      saidas [evalcruzada],
                      batch_size = 128, # Pesos são atualizados a cada 128 amostras
                      epochs = 5) # executa a rede 10x
    precisao = classificador.evaluate(entradas[svalcruzada], saidas [svalcruzada])
    resultados.append(precisao[1])
                
media = sum(resultados) / len(resultados)