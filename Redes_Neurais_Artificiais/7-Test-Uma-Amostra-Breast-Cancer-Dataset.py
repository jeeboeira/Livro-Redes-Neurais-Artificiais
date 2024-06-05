# Repositório dos dados breast cancer
from ucimlrepo import fetch_ucirepo
import pandas as pd
#Converte os dados para um mesmo parâmetro
from sklearn.preprocessing import LabelEncoder
# Modelo pré configurado que cria e faz as conexões entre camadas de neurônios
from keras.models import Sequential
#Conecta todos os nós da rede e aplica fuções em todas as etapas
from keras.layers import Dense, Dropout

# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes) 
entradas = breast_cancer_wisconsin_diagnostic.data.features
#.values.ravel() deixa a saída pronta para labelEncoder()
saidas = breast_cancer_wisconsin_diagnostic.data.targets.values.ravel()

lb = LabelEncoder()

saidasLb = lb.fit_transform(saidas)
saidasDf = pd.DataFrame(saidasLb) 


# Cria a rede neural
classificadorA = Sequential()

#Crio essa rede segundo os parâmetros do Tunning
#Cria a primeira camada oculta através da função add Dense
classificadorA.add(Dense(units = 8, # units = quantidade de neurônios na camada oculta
                         activation = 'relu', # Rectfied Linear Units - algo próximo ao StepFunction
                         kernel_initializer = 'random_uniform',
                         input_dim = 30)) # Quantos parâmetros tem nossa camada de entrada
classificadorA.add(Dropout(0.2))
# Aqui não é necessário especificar o dim, pois já é implícito
classificadorA.add(Dense(units = 8,
                         activation = 'relu',
                         kernel_initializer = 'random_uniform'))
classificadorA.add(Dropout(0.2))
classificadorA.add(Dense(units = 1, #camada de saída
                        activation = 'sigmoid'))
#Compilador
classificadorA.compile(optimizer = 'adam',
                      loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
# Alimenta a rede e define os parâmetros de processamento
classificadorA.fit(entradas,
                   saidasDf,
                   batch_size = 10, # divide a rede em 10 para teste
                   epochs = 50) # Roda os teste 50 vezes

#Objeto selecionado aleatóriamente da base de dados para teste
objeto = entradas.iloc[68].to_frame()
objetoT = objeto.T

previsorA = classificadorA.predict(objetoT)
previsorB = (previsorA > 0.5)


