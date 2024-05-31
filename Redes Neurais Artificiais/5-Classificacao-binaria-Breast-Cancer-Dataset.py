import pandas as pd
import numpy as np
# Repositório dos dados breast cancer
from ucimlrepo import fetch_ucirepo 
#
from sklearn.model_selection import train_test_split
#
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
#Converte os dados para um mesmo parâmetro
from sklearn.preprocessing import LabelEncoder
#
from sklearn.metrics import confusion_matrix, accuracy_score
# Uso para criar saidasDummy para trabalhar com regressão numérica
from keras.utils import np_utils


# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
# metadata 
#print(breast_cancer_wisconsin_diagnostic.metadata) 
# variable information 
#print(breast_cancer_wisconsin_diagnostic.variables) 


# data (as pandas dataframes) 
entradas = breast_cancer_wisconsin_diagnostic.data.features 
saidas = breast_cancer_wisconsin_diagnostic.data.targets.values

lb = LabelEncoder()

saidasLb = lb.fit_transform(saidas)

# Divisão da base para teste e certificação da integridade
etreino, eteste, streino, steste = train_test_split(entradas,
                                                    saidasLb,
                                                    test_size = 0.25)


# Cria a rede neural
classificador = Sequential()

#Cria a primeira camada oculta através da função add Dense
classificador.add(Dense(units = 16, # # units = quantidade de neurônios na camada oculta
                        activation = 'relu', # Rectfied Linear Units - algo próximo ao StepFunction
                        kernel_initializer = 'random_uniform',
                        input_dim = 30)) # Quantos parâmetros tem nossa camada de entrada

# Segunda camada oculta
# Aqui não é necessário especificar o dim, pois já é implícito
classificador.add(Dense(units = 1, #camada de saída
                        activation = 'sigmoid'))

#Compilador
classificador.compile(optimizer = 'adam',
                      loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

# Alimenta a rede e define os parâmetros de processamento
classificador.fit(etreino,
                  streino,
                  batch_size = 10, # divide a rede em 10 para teste
                  epochs = 100) # Roda os teste 100 vezes


## Avaliação de performance e integridade dos resultados
previsor = classificador.predict(eteste)

steste2 = [np.argmax(t) for t in steste]
previsoes2 = [np.argmax(t) for t in previsor]


margem_acertos = accuracy_score(steste, previsor)
matriz_confusão = confusion_matrix(steste, previsor)
resultadoMatrizConfusao = classificador.evaluate(eteste, steste)