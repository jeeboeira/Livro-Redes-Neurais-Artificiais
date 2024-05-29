import pandas as pd
import numpy as np
#Converte os dados para um mesmo parâmetro
from sklearn.preprocessing import LabelEncoder
#
from keras.utils import np_utils
# Divide os dados para validação mais acertiva
from sklearn.model_selection import train_test_split
# Modelo pré configurado que cria e faz as conexões entre camadas de neurônios
from keras.models import Sequential
#Conecta todos os nós da rede e aplica fuções em todas as etapas
from keras.layers import Dense
#
import numpy as np
# Gerar matrix de confusão
from sklearn.metrics import confusion_matrix
# Fazer validação cruzada
from sklearn.model_selection import cross_val_score
#Alimenta nossa validação cruzada
from keras.wrappers.scikit_learn import KerasClassifier
#from scikeras.wrappers import KerasClassifier

#Importa os dados da planilha para a variável base
#base = pd.read_csv('base\iris.csv')

#Base para rápido envio para spyder
base = pd.read_csv(r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\base\iris.csv')


# Pega todos os valores contidos em todas as linhas e das 4 primeiras
    #colunas de base (desconsidera index)
entradas = base.iloc[:, 0:4].values
saidas = base.iloc[:, 4].values

#Inicializa o Label Encoder
labelencoder = LabelEncoder()

#Transforma os dados de saída de string para int
saidas = labelencoder.fit_transform(saidas)


#Crio variáveis do tipo dummy para indexação da saída
    #Ou seja, quais neurônios ligam para cada saída
    #Basicamente cada neuronio sai 0 ou 1
    #Eu atribuo um valor binário, que é a soma dos neurônios ativados

#Cria números de indexação para cada uma das amostras da base de dados
saidas_dummy = np_utils.to_categorical(saidas)

#Crio as variáveis para divisão das amostras e seto 0.25, ou seja:
    #25% para treino e 75% para teste
etreino, eteste, streino, steste = train_test_split(entradas,
                                                    saidas_dummy,
                                                    test_size = 0.25)


# Inicializa Sequential
classificador = Sequential()

#Cria a primeira camada oculta através da função add Dense
classificador.add(Dense(units = 4,           # units = quantidade de neurônios
                        activation = 'relu', # Rectfied Linear Units - algo próximo ao StepFunction
                        input_dim = 4))      # Quantos parâmetros tem nossa camada de entrada

# Segunda camada oculta
# Aqui não é necessário especificar o dim, pois já é implícito
classificador.add(Dense(units = 4,
                        activation = 'relu'))

# Camada de saída
classificador.add(Dense(units = 3,
                        activation = 'softmax')) # Função similar a sigmoide

#Compilador
classificador.compile(optimizer = 'adam',                   # Descida do Gradiente Estocástico
                      loss = 'categorical_crossentropy',    # Função de perda
                      metrics = ['categorical_accuracy'])   # Avaliação interna do modelo

# Alimenta a rede e define os parâmetros de processamento
classificador.fit(etreino,
                  streino,
                  batch_size = 10, # Taxa de atualização dos pesos
                  epochs = 1000)   # Vezes que a rede será executada

# Avaliação de performance e integridade dos resultados
avalPerformance = classificador.evaluate(eteste, steste)

previsoes = classificador.predict(eteste)
previsoesVF = (previsoes > 0.5)

# Matriz de confusão
# np.argmax(t) pega os valores de steste e gera uma indexação própria com a posição da matriz onde a planta foi classificada
    # O mesmo é feito em previsoes2
steste2 = [np.argmax(t) for t in steste]
previsoes2 = [np.argmax(t) for t in previsoes]

matrizConfusao = confusion_matrix(previsoes2, steste2)

# Fazer a validação cruzada
    # Divide os dados em partes iguais e checa se os resultados correspondem
def valCruzada():
    classificadorValCruzada = Sequential()
    classificadorValCruzada.add(Dense(units = 4,
                                      activation = 'relu',
                                      input_dim = 4))
    classificadorValCruzada.add(Dense(units = 4,
                                      activation = 'relu'))
    classificadorValCruzada.add(Dense(units = 3,
                                      activation = 'softmax'))
    classificadorValCruzada.compile(optimizer = 'adam',
                                    loss = 'categorical_crossentropy',
                                    metrics = ['categorical_accuracy'])
    return classificadorValCruzada


# Variável que executa KerasClassifier, e recebe a função valCruzada
classificadorValCruzada = KerasClassifier(build_fn = valCruzada,
                                          epochs = 1000,
                                          batch_size = 10) # Taxa de correção de pesos


#Divide o processamento em várias partes segundo o cv e as executa
resultadosValCruzada = cross_val_score(estimator = classificadorValCruzada,
                                       X = entradas,
                                       y = saidas,
                                       cv = 10, #Define em quantas partes minha base sera dividida
                                                    #igualmente para ser processada
                                       scoring = 'accuracy')

#
media = resultadosValCruzada.mean()
desvio = resultadosValCruzada.std()