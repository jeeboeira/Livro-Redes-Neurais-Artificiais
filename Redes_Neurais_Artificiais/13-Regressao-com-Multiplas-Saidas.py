import pandas as pd
#Dense Cria as camadas da rede
from keras.layers import Dense, Input
# Model permite criar uma rede densa com múltiplas saídas
from keras.models import Model
#Converte os dados para um mesmo parâmetro
from sklearn.preprocessing import LabelEncoder
#Indexa os valores para o meu interpretador compreender
from sklearn.preprocessing import OneHotEncoder
#Funciona junto com o onehotecoder
from sklearn.compose import ColumnTransformer

#importa o dataset
base = pd.read_csv(r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_Artificiais\Arquivos\Treze\vgsales.csv')

#Variável para eu acompanhar as mudanças
abaseOriginal = base

#POLIMENTO DOS DADOS

# Imprime as 5 primeiras linhas
print (base.head())
#Mostra a quantidade de dados e colunas
print (base.shape)

#Dropa as colunas desnecessárias
base = base.drop(['Other_Sales', 'Global_Sales'], axis = 1)
# Remove dados faltante nas linhas
base = base.dropna (axis = 0)
# Mantém somente os dados com valores maiores que 1
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

#Salva a coluna em uma variável antes de dropar
#.value_counts() mostra quantas vezes tem cada valor
base['Name'].value_counts()
backupName = base.Name
base = base.drop('Name', axis = 1)

#Separa as entradas e saídas
entradas = base.iloc[:, [0,1,2,3,4]].values
vendas_NA = base.iloc[:, 5].values
vendas_EU = base.iloc[:, 6].values
vendas_JP = base.iloc[:, 7].values


#Transforma todos os dados para numéricos
lb = LabelEncoder()

entradas[:,1] = lb.fit_transform(entradas[:,1])
entradas[:,2] = lb.fit_transform(entradas[:,2])
entradas[:,3] = lb.fit_transform(entradas[:,3])
entradas[:,4] = lb.fit_transform(entradas[:,4])

#Transforma os dados para modelo categórico indexado
ct = ColumnTransformer([("", OneHotEncoder(), [1,2,3,4])], remainder='passthrough')
entradas = ct.fit_transform(entradas).toarray()


#FIM DA FASE DE POLIMENTO


# Cria a rede neural
#Essa rede ainda sequêncial, mas que tem múltiplas saídas independentes
    # Para ter comunicação entre uma cada e outra, a referência é passada
    # Como segundo parâmetro independente
camada_entrada = Input(shape = (90, ))
camada_oculta1 = Dense(units = 45,
                       activation = 'sigmoid')(camada_entrada)
camada_oculta2 = Dense(units = 45,
                       activation = 'sigmoid')(camada_oculta1)
camada_saida1 = Dense(units = 1,
                      activation = 'linear')(camada_oculta2)
camada_saida2 = Dense(units = 1,
                      activation = 'linear')(camada_oculta2)
camada_saida3 = Dense(units = 1,
                      activation = 'linear')(camada_oculta2)
    #Função linear não aplica nenhuma função, só replica o valor encontrado

# Cria Regressor com o atributo Model
regressor = Model(inputs = camada_entrada,
                  outputs = [camada_saida1,
                             camada_saida2,
                             camada_saida3])
regressor.compile(optimizer = "adam",
                  loss = 'mse')#Mean Squared Error, um modelo de função de perda
regressor.fit(entradas,
              [vendas_NA, vendas_EU, vendas_JP],
              epochs = 5000, # Roda a rede 5000 vezes
              batch_size = 100) # Atualiza os dados a cada 100 amostras

# Previsor
previsao_NA, previsao_EU, previsao_JP = regressor.predict(entradas)