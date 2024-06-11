import pandas as pd
import numpy as np
# Parecido com MathPLotLib exibe os dados de forma visual
import seaborn as sns
#Divide a base de dados e testa partes individuais para validar
from sklearn.model_selection import train_test_split
#Faz regressão Linear
from sklearn.linear_model import LinearRegression
#Avalia a performance do modelo
from sklearn import metrics


#importa o dataset
base = pd.read_csv(r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_Artificiais\Arquivos\Quatorze\Advertising.csv')

#Variável para eu acompanhar as mudanças
abaseOriginal = base

#POLIMENTO DOS DADOS

# Imprime as 5 primeiras linhas
print (base.head())
#Mostra a quantidade de dados e colunas
print (base.shape)

#Cria uma variável que cruza os dados
sns = sns.pairplot(base,
                   x_vars = ['TV', 'radio', 'newspaper'],
                   y_vars = 'sales',
                   height = 5,
                   kind = 'reg') # Dados serão apresentados em forma de regressão
#Aqui o que há de mais interessante é a slinha ascendente, que mostra que
    #houve retorno positivo dos investimentos


# Pega todos os valores contidos em todas as linhas e das 4 primeiras
    #colunas de base (desconsidera index)
entradas = base.iloc[:, 1:4].values
saidas = base.iloc[:, 4].values

# Aqui uso train test split para avaliar a performance do modelo
etreino, eteste, streino, steste = train_test_split(entradas,
                                                    saidas,
                                                    test_size = 0.3)#Divide em 30% para teste 70% para treino

#executa a regressão
reglinear = LinearRegression()
reglinear.fit(etreino, streino)

#reglinear.coef_ ratorna o valor de coeficiente (de investimento nesse caso)
print(list(zip(['TV', 'radio', 'newspaper'], reglinear.coef_)))
# O retorno pode ser interpretado como valores de unidade monetária
    #Ex.: "TV" retorna 0.04 ou seja, para cada dólar, houve um aumento de 
    #4% nas vendas

# Aqui passo uma linha da minha base, e chamo predict
print(reglinear.predict([[230.1, 37.8, 69.2]]))
#O valor retornado de 20.51 mostra que em uma campanha investido
    #230 em tv, 37 em radio e 69 em jornal, teve um retorno de 20%

# Mostra o retorno para cada linha de campanha
previsor = reglinear.predict(eteste)
print(previsor)

#Confirma a integridade do Modelo

# Nesse caso, a saída dela, mostra o retorno para cada dólar investido
mae = metrics.mean_absolute_error(steste, previsor)
# No caso a cada dolar, volta uma média de 1,30 dólares

# Aqui os valores de erro são elevados ao quadrado, dando mais peso aos erros
mse = metrics.mean_squared_error(steste, previsor)
#Um erro 2 ao quadrado é 4 já 5 é 25, o que gera grande impacto nas funções
    #porém esse valor não deve ser convertido diretamente para valor monetário
    #primeiro é necessário aplicar outra função

#Aplica a raiz quadrada sobre mse
rmse = np.sqrt(metrics.mean_squared_error(steste,previsor))
#rmse = np.sqrt(mse) É a mesma coisa
#Agora temos um retorno por dólar
