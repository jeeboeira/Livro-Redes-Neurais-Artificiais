import pandas as pd
#
from sklearn.model_selection import train_test_split
# Modelo pré configurado que cria e faz as conexões entre camadas de neurônios
from keras.models import Sequential
#Conecta todos os nós da rede e aplica fuções em todas as etapas
from keras.layers import Dense, Dropout
#Converte os dados para um mesmo parâmetro
from sklearn.preprocessing import LabelEncoder
#Faz a matriz de confusão e score de acurácia
from sklearn.metrics import confusion_matrix, accuracy_score
#Validação Cruzada
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# Repositório dos dados breast cancer
from ucimlrepo import fetch_ucirepo 


# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
# metadata 
#print(breast_cancer_wisconsin_diagnostic.metadata) 
# variable information 
#print(breast_cancer_wisconsin_diagnostic.variables) 


# data (as pandas dataframes) 
entradas = breast_cancer_wisconsin_diagnostic.data.features
#.values.ravel() deixa a saída pronta para labelEncoder()
saidas = breast_cancer_wisconsin_diagnostic.data.targets.values.ravel()

lb = LabelEncoder()

saidasLb = lb.fit_transform(saidas)
saidasDf = pd.DataFrame(saidasLb)

# Divisão da base para teste e certificação da integridade
etreino, eteste, streino, steste = train_test_split(entradas,
                                                    saidasDf,
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
                  epochs = 1) # Roda os teste 100 vezes


## Avaliação de performance e integridade dos resultados
previsor = classificador.predict(eteste)
previsorVF = (previsor > 0.5)

margem_acertos = accuracy_score(steste, previsorVF)
matriz_confusão = confusion_matrix(steste, previsorVF)
resultadoMatrizConfusao = classificador.evaluate(eteste, steste)


#Validação Cruzada
    # Divide os dados em partes iguais e checa se os resultados correspondem
def valCruzada():
    classificadorValCruzada = Sequential()
    #Cria a primeira camada oculta através da função add Dense
    classificadorValCruzada.add(Dense(units = 16, # # units = quantidade de neurônios na camada oculta
                                activation = 'relu', # Rectfied Linear Units - algo próximo ao StepFunction
                                kernel_initializer = 'random_uniform',
                                input_dim = 30)) # Quantos parâmetros tem nossa camada de entrada
    #Exclui 20% dos dados para o sistema trabalhar
    classificadorValCruzada.add(Dropout(0.2))
    # Aqui não é necessário especificar o dim, pois já é implícito
    classificadorValCruzada.add(Dense(units = 16, # # units = quantidade de neurônios na camada oculta
                                activation = 'relu', # Rectfied Linear Units - algo próximo ao StepFunction
                                kernel_initializer = 'random_uniform'))
    # Segunda camada oculta
    classificadorValCruzada.add(Dense(units = 1, #camada de saída
                                activation = 'sigmoid'))
    #Compilador
    classificadorValCruzada.compile(optimizer = 'adam',
                                    loss = 'binary_crossentropy',
                                    metrics = ['binary_accuracy'])
    return classificadorValCruzada


# Variável que executa KerasClassifier, e recebe a função valCruzada
ClassificadorValCruzada = KerasClassifier(build_fn = valCruzada,
                                          epochs = 1,
                                          batch_size = 10)

#Divide o processamento em várias partes segundo o cv e as executa
resultadoValCruzada = cross_val_score(estimator = ClassificadorValCruzada,
                                      X = entradas,
                                      y = saidas,
                                      cv = 10, #Define em quantas partes minha base sera dividida
                                                    #igualmente para ser processada
                                      scoring = 'accuracy')

#
mediaValCruzada = resultadoValCruzada.mean()
desvioValCruzada = mediaValCruzada.std()