# Repositório dos dados breast cancer
from ucimlrepo import fetch_ucirepo
# Biblioteca Matricial
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
classificadorB = Sequential()

#Crio essa rede segundo os parâmetros do Tunning
#Cria a primeira camada oculta através da função add Dense
classificadorB.add(Dense(units = 8, # units = quantidade de neurônios na camada oculta
                         activation = 'relu', # Rectfied Linear Units - algo próximo ao StepFunction
                         kernel_initializer = 'random_uniform',
                         input_dim = 30)) # Quantos parâmetros tem nossa camada de entrada
classificadorB.add(Dropout(0.2))
# Aqui não é necessário especificar o dim, pois já é implícito
classificadorB.add(Dense(units = 8,
                         activation = 'relu',
                         kernel_initializer = 'random_uniform'))
classificadorB.add(Dropout(0.2))
classificadorB.add(Dense(units = 1, #camada de saída
                        activation = 'sigmoid'))
#Compilador
classificadorB.compile(optimizer = 'adam',
                      loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
# Alimenta a rede e define os parâmetros de processamento
classificadorB.fit(entradas,
                   saidasDf,
                   batch_size = 10, # divide a rede em 10 para teste
                   epochs = 50) # Roda os teste 100 vezes


#SALVANDO O MODELO
#Criado variável que recebe toda a estrutura da rende neural
    #classificadorB com a função .to_json()
classificador_json = classificadorB.to_json()

#Cria e escreve no arquivo
with open('Redes Neurais Artificiais\Arquivos\Oito\classificador_binario.json', 'w') as json_file:
    json_file.write(classificador_json)

#Usando função do pandas, guarda os pesos do código
classificadorB.save_weights('Redes Neurais Artificiais\Arquivos\Oito\classificador_binario_pesos.h5')

