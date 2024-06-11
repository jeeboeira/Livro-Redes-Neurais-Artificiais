import pandas as pd
#Converte os dados para um mesmo parâmetro
from sklearn.preprocessing import LabelEncoder
#Indexa os valores para o meu interpretador compreender
from sklearn.preprocessing import OneHotEncoder
#Funciona junto com o onehotecoder
from sklearn.compose import ColumnTransformer
# Modelo pré configurado que cria e faz as conexões entre camadas de neurônios
from keras.models import Sequential
#Conecta todos os nós da rede e aplica fuções em todas as etapas
from keras.layers import Dense
#Validação Cruzada
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor


#importado o dataset, com encoding padrão
base = pd.read_csv(r"C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_Artificiais\Arquivos\Onze\autos.csv", encoding = "ISO-8859-1")

#Variável para eu acompanhar as mudanças
abaseOriginal = base

# Imprime as 5 primeiras linhas
#print(base.head())

#Exclui as colunas pelo nome
base = base.drop(['index', 'dateCrawled', 'dateCreated', 'nrOfPictures', 'postalCode', 'lastSeen'], axis = 1)

#Aqui eu vejo o número de ocorrência de cada valor, como existem muitas inconsistencias, melhor dropar
#print(base['name'].value_counts())
base = base.drop(['name', 'seller', 'offerType'], axis = 1)

#Aqui filtro amostras com valores absurdamente baixos
varTeste1 = base.loc[base.price <=10]
#Assim faço o método ao contrário, e salvo na mesma variável
base = base[base.price > 10]

#Aqui filtro os valores muito acima
varTeste2 = base.loc[base.price > 350000]
#Assim faço o método ao contrário, e salvo na mesma variável
base = base[base.price < 350000]

# Utilizando o método .loc da função .isnull eu limpo os dados faltantes.
base.loc[pd.isnull(base['vehicleType'])]
#Faz o agrupamento de dados, com ele pego o veículo com mais amostras para preencher as celulas que tem inconsistencias, sem excluí-las
base['vehicleType'].value_counts()

#Repito o processo para todas as colunas com dados faltantes
#Esse eu passo a coluna, e ele me mostra somente os nulos da coluna
base.loc[pd.isnull(base['gearbox'])]
#Esse me mostra os valores atribuídos, com as quantidades de recorrencia de cada um
base['gearbox'].value_counts()
base.loc[pd.isnull(base['model'])]
base['model'].value_counts()
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts()
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts()

# Crio um dicionário com os parâmetros e so valores que vou atribuir
valores_medios = {'vehicleType': 'limousine',
                  'gearbox': 'manuell',
                  'model': 'golf',
                  'fuelType': 'benzin',
                  'notRepairedDamage': '259301'}

#Fillna preenche todos os valores NA/NaN
base = base.fillna(value = valores_medios)

#Entradas recebe todas as colunas de 1 até 13
entradas = base.iloc[:, 1:13].values
#Saída recebe os dados da coluna 0, price
saidas = base.iloc[:, 0].values

#Muda os valores para números para serem pocessados pela rede neural
lb = LabelEncoder()

entradas[:,0] = lb.fit_transform(entradas[:,0])
entradas[:,1] = lb.fit_transform(entradas[:,1])
entradas[:,3] = lb.fit_transform(entradas[:,3])
entradas[:,5] = lb.fit_transform(entradas[:,5])
entradas[:,8] = lb.fit_transform(entradas[:,8])
entradas[:,9] = lb.fit_transform(entradas[:,9])
entradas[:,10] = lb.fit_transform(entradas[:,10])

#Transforma os dados para modelo categórico indexado
ct = ColumnTransformer([("", OneHotEncoder(), [0,1,3,5,8,9,10])], remainder='passthrough')
entradas = ct.fit_transform(entradas).toarray()

#FIM DA FASE DE POLIMENTO


# Cria a rede neural
#Esta vez não foi criado um classificador mas sim um Regressor
regressor = Sequential()

#Cria a primeira camada ou camada de entrada através da função add.
regressor.add(Dense(units = 158, # units = quantidade de neurônios na camada oculta
                    activation = 'relu', # Rectfied Linear Units - algo próximo ao StepFunction
                    input_dim = 317)) # Quantos parâmetros tem nossa camada de entrada
# Aqui não é necessário especificar o dim, pois já é implícito
regressor.add(Dense(units = 158,
                    activation = 'relu'))
regressor.add(Dense(units = 1, #camada de saída
                    activation = 'linear'))
#Compilador
regressor.compile(optimizer = 'adam',
                  loss = 'mean_absolute_error',
                  metrics = ['mean_absolute_error'])

#Esta vez não foi criado um classificador mas sim um 

# Alimenta a rede e define os parâmetros de processamento
regressor.fit(entradas,
              saidas,
              batch_size = 300, # divide a rede em 300 para teste
              epochs = 100) # Roda os teste 100 vezes

#Previsões do que tem que acontecer
previsoes = regressor.predict(entradas)
saidas.mean()
previsoes.mean()
saidaMean = saidas.mean()
previsoesMean = previsoes.mean()


#Validação Cruzada


def regressorValCruzada():
    regressorV = Sequential()
    regressorV.add(Dense(units = 158, # units = quantidade de neurônios na camada oculta
                         activation = 'relu', # Rectfied Linear Units - algo próximo ao StepFunction
                         input_dim = 317)) # Quantos parâmetros tem nossa camada de entrada
    # Aqui não é necessário especificar o dim, pois já é implícito
    regressorV.add(Dense(units = 158,
                         activation = 'relu'))
    regressorV.add(Dense(units = 1, #camada de saída
                         activation = 'linear'))
    #Compilador
    regressorV.compile(optimizer = 'adam',
                       loss = 'mean_absolute_error',
                       metrics = ['mean_absolute_error'])
    return regressorV

# Executa o regressorV
regValCruzada = KerasRegressor(build_fn = regressorValCruzada,
                               epochs = 100,
                               batch_size = 300)# Taxa de correção de pesos

resValCruzada = cross_val_score(estimator = regValCruzada,
                                X = entradas,
                                y = saidas,
                                cv = 10, #Define em quantas partes minha base sera dividida
                                                    #igualmente para ser processada
                                scoring = 'neg_mean_absolute_error') # Média dos resultados desconsiderando o sinal

media = resValCruzada.mean()
desvioPadrao = resValCruzada.std()