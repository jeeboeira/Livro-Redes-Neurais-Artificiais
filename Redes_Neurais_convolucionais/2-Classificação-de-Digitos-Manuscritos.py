import matplotlib.pyplot as plt
from keras.datasets import mnist
# Modelo pré configurado que cria e faz as conexões entre camadas de neurônios
from keras.models import Sequential
#Dense Cria as camadas da rede
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
#Conv2D cria camadas adaptadas ao processamento de imagens
#MaxPooling2D cria micromatrizes indexadas, para identificação
    #de padrões pixel a pixel
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import batch_normalization

(etreino, streino),(eteste, steste) = mnist.load_data()

#Plota
plt.imshow(etreino[0], cmap = 'gray')
plt.title('Classe' + str(streino[0]))

#FASE DE POLIMENTO

etreino = etreino.reshape(etreino.shape[0],28,28,1)
eteste = eteste.reshape(eteste.shape[0],28,28,1)

etreino = etreino.astype('float32')
eteste = eteste.astype('float32')

etreino /= 255
eteste /= 255

#Classifico as saídas de 0 a 9
streino = np_utils.to_categorical(streino, 10)
steste = np_utils.to_categorical(steste, 10)

#FIM DO POLIMENTO

classificador = Sequential()
classificador.add(Conv2D(32,
                         (3,3),
                         input_shape = (28,28,1),
                         activation = 'relu'))
classificador.add(batch_normalization())#Faz uma limpeza de pixels borrados
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Conv2D(32,
                         (3,3),
                         activation = 'relu'))
classificador.add(batch_normalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
#Transforma as matrizes para uma dimensão, para cada valor da linha atuar como um neurônio
classificador.add(Flatten())
#Cria camadas densas
classificador.add(Dense(units = 128,
                        activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128,
                        activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, # Camada de saida com 10 neurônios, já que a classificação
                                        #é de 0 a 9.
                        activation = 'softmax'))#Parecido com sigmoid, porém é capaz
                                        #de categorizar amostras em múltiplas saídas

# COMPILADOR

classificador.compile(loss = 'categorical_crossentropy', # Verifica quais dados não foram possíveis
                      #classificar quanto sua proximidade aos vizinhos.
                      optimizer = 'adam',
                      metrics = ['accuracy'])
classificador.fit(etreino, # Alimenta a rede
                  streino,
                  batch_size = 128, # Pesos são atualizados a cada 128 amostras
                  epochs = 10, # executa a rede 10x
                  validation_data = (eteste, steste))