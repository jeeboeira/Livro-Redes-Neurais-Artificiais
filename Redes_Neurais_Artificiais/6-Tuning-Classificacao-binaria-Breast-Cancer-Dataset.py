# Repositório dos dados breast cancer
from ucimlrepo import fetch_ucirepo
import pandas as pd
#Converte os dados para um mesmo parâmetro
from sklearn.preprocessing import LabelEncoder
# Modelo pré configurado que cria e faz as conexões entre camadas de neurônios
from keras.models import Sequential
#Conecta todos os nós da rede e aplica fuções em todas as etapas
from keras.layers import Dense, Dropout
#Alimenta nossa validação cruzada
from keras.wrappers.scikit_learn import KerasClassifier
#Cria parâmetros manuais para alimentar a rede
from sklearn.model_selection import GridSearchCV

# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes) 
entradas = breast_cancer_wisconsin_diagnostic.data.features
#.values.ravel() deixa a saída pronta para labelEncoder()
saidas = breast_cancer_wisconsin_diagnostic.data.targets.values.ravel()

lb = LabelEncoder()

saidasLb = lb.fit_transform(saidas)
saidasDf = pd.DataFrame(saidasLb)

#Defino a função com os parâmetros manuais
def tuningClassificador(optimizer, loss, kernel_initializer, activation, neurons):
    classificadorTuning = Sequential()
    classificadorTuning.add(Dense(units = neurons,
                                  activation = activation,
                                  kernel_initializer = kernel_initializer,
                                  input_dim = 30))
    classificadorTuning.add(Dropout(0.2))
    classificadorTuning.add(Dense(units = neurons,
                                  activation = activation,
                                  kernel_initializer = kernel_initializer))
    classificadorTuning.add(Dropout(0.2))
    classificadorTuning.add(Dense(units = 1,
                                  activation = 'sigmoid'))
    classificadorTuning.compile(optimizer = optimizer,
                                loss = loss,
                                metrics = ['binary_accuracy'])
    return classificadorTuning

#
classificadorTunado = KerasClassifier(build_fn = tuningClassificador)


# variável que recebe em forma de dicionário os parâmetros de testes
    #ex.: ele usará 'relu' em um teste e 'tanh' em outro, posteriormente,
    #a GridSearchCV irá verificar qual parâmetro obteve melhor retorno.
parametros = {'batch_size':[10,30],
              'epochs':[50,100],
              'optimizer':['adam', 'sgd'],
              'loss':['binary_crossentropy', 'hinge'],
              'kernel_initializer':['random_uniform','normal'],
              'activation':['relu','tanh'],
              'neurons':[10,8]}

#
tunagem = GridSearchCV(estimator = classificadorTunado,
                       param_grid = parametros,
                       scoring = 'accuracy')
# Aplica em si a função fit, que dará início a execução da rede neural
tunagem = tunagem.fit(entradas,saidasDf)

#
melhores_parametros = tunagem.best_params_
melhor_margem_precisao = tunagem.best_score_