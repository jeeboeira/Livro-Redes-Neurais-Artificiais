import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

#Importo os dados csv
data = pd.read_csv (r"C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\base\iris.csv")

#Faço o polimento dos dados
data = data.drop(['class', 'petal lenght', 'petal width'], axis = 1)
data = data.rename({'sepal length' : 'X', 'sepal width':'y'}, axis = 1)

#Ploto um gráfico de dispersão
plt.scatter(data['X'], data['y'])
plt.show()

#
modelo = tf.keras.Sequential()
#Meu dense recebe 1 neurônio de saída e 1 neurônio de entrada
modelo.add(tf.keras.layers.Dense(1, input_shape=[1]))
modelo.compile(loss = 'mean_squared_error', # A cada erro, eleva o valor ao quadrado
               optimizer = tf.keras.optimizers.Adam(0.01)) # O reajuste de peso é realizado a cada 1% dos dados processados
#Me mostra um sumário dos dados
print(modelo.summary())
#Roda a rede neural
modelo.fit(data['X'], data['y'], epochs = 1000)

#
data['Previsao'] = modelo.predict(data['X'])

#Plota os dados, com a previsão em relação a coluna X, e apresenta o melhor ponto de equivalência entre os pontos
plt.scatter(data['X'], data['y'])
plt.plot(data['X'], data['Previsao'], color = 'r')
plt.show()