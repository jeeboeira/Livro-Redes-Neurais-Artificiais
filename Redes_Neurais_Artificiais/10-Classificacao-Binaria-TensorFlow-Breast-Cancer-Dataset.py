# Do tensorflow, vou usar o núcleo e sub-bibliotecas no caso Keras para criação do modelo
# Keras para criação do modelo
import tensorflow as tf
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv1D, MaxPool1D, BatchNormalization
from keras.optimizers import Adam
# Pandas e Numpy para manipulação de dados matriciais 
import pandas as pd
import numpy as np
#Sklearn para o tratamento dos dados
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#Matplot irá gerar os gráficos
import matplotlib.pyplot as plt

# Insere o banco de dados em uma variável
data = datasets.load_breast_cancer()

#Printa toda a descrição do banco de dados
print(data.DESCR)

#salva um dataframe de entradas na variável X
X = pd. DataFrame(data = data.data,
                  columns = data.feature_names)
print(X.head())

# Salva as saídas esperadas na variável y
y = data.target
print(y)

#mostra o nome das saídas possíveis
print(data.target_names)

#printa o formato do DataFrame X
print(X.shape)

X_treino, X_teste, y_treino, y_teste = train_test_split(X,
                                                        y,
                                                        #80% dos dados são para treino e
                                                            #20% são para teste
                                                        test_size = 0.2,
                                                        #random_state randomiza as amostras para não ter vício
                                                            #como meu dataset é pequeno, não preciso.
                                                        random_state = 0,
                                                        #força a divisão proporcional, utilizado somente em classificações binárias
                                                        stratify = y)
# Tamapnho de cada parte
print(X_treino.shape)
print(X_teste.shape)


#Deixa os dados de entrada em valores na escala de 0 à 1
escalonador = StandardScaler()

X_treino = escalonador.fit_transform(X_treino)
X_teste = escalonador.transform(X_teste)

#Transforma o formato dos dados deixando somente uma dimensão
X_treino = X_treino.reshape(455, 30, 1)
X_teste = X_teste.reshape(114, 30, 1)


# Cria a rede neural
modelo = Sequential()

#Cria a primeira camada ou camada de entrada através da função add.
    #Foi utilizado Conv1D pois os dados foram obtidos e mapeados
    #a partir de imagens.
modelo.add(Conv1D(filters = 32, # Filtros aplicados a cada amostrano processo interno de convolução
                  kernel_size = 2, # Divide a imagem em blocos e mapeia as caracteristicas
                  activation = 'relu', # Rectfied Linear Units - algo próximo ao StepFunction
                  input_shape = (30,1))) # Número de neurônios da camada de entrada
#
modelo.add(BatchNormalization())
# Desliga aleatóriamente alguns neurônios
modelo.add(Dropout(0.2))
# Camada oculta
modelo.add(Conv1D(filters = 64,
                  kernel_size = 2,
                  activation = 'relu'))
#           
modelo.add(BatchNormalization())
modelo.add(Dropout(0.5))

#Redimensiona os dados
modelo.add(Flatten())
# Camada oculta agora com neurônios sem convoluções
modelo.add(Dense(64,
                 activation = 'relu'))
modelo.add(Dropout (0.5))

#
modelo.add(Dense(1,
                 activation = 'sigmoid'))

#Mostra a estrutura da rede
modelo.summary()

#Compila os dados
modelo.compile(optimizer = Adam(learning_rate = 0.00005), # Taxa de aprendizado
               loss = 'binary_crossentropy', # Parâmetro para identificação dos erros
               metrics =['accuracy']) # métrica de aprendizado, no caso baseado em precisão

epochs=1000
history = modelo.fit(X_treino,
                     y_treino,
                     epochs = epochs,
                     validation_data = (X_teste, y_teste),
                     verbose = 1) # Exibe o resultado do processamento

#Impressão de gráficos da aprendizagem
def CurvaAprendizado(historico, epoca):
    epoca = range(1, epoca + 1)
    plt.plot(epoca, historico.history['accuracy'])
    plt.plot(epoca, historico.history['val_accuracy'])
    plt.title('Precisão do Modelo')
    plt.ylabel('Margem de Acertos')
    plt.xlabel('Épocas')
    plt.legend(['Train', 'Val'], loc = 'upper left')
    plt.show()

    plt.plot(epoca, historico.history['loss'])
    plt.plot(epoca, historico.history['val_loss'])
    plt.title('Margem de Erro')
    plt.ylabel('Erros')
    plt.xlabel('Épocas')
    plt.legend(['Train', 'Val'], loc = 'upper left')
    plt.show()
CurvaAprendizado(history, epochs)