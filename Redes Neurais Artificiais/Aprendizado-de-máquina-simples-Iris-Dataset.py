from sklearn.datasets import load_iris
#Módulo knn importado da biblioteca sklearn
from sklearn.neighbors import KNeighborsClassifier
#Divide a base de dados e testa partes individuais para validar
from sklearn.model_selection import train_test_split
#Avalia a performance do modelo
from sklearn import metrics
#Cria gráficos dos meus dados
import matplotlib.pyplot as plt
#Faz regressão Logística, probabilidade da amostra fazer parte de um tipo, ou outro
from sklearn.linear_model import LogisticRegression


base = load_iris()

print(base.data)
print(base.target)
print(base.target_names)

entradas = base.data
saidas = base.target
rotulos = base.target_names

print(base.data.shape)
print(base.target.shape)

# Variável que classifica a proximidade de relação com a amostra vizinha, no caso o parâmetro é 1
knn = KNeighborsClassifier(n_neighbors = 1)
# Alimenta o classificador com os dados a serem processados
knn.fit(entradas, saidas)
#Realiza uma simples previsão, pegando os dados de uma linha na base de dados
knn.predict([[5.1,3.1,1.4,0.2]])

especie = knn.predict([[5.9,3,5.1,1.8]])[0]
print(especie)
rotulos[especie]

#variáveis para as amostras de treino
# test_size aqui definido 0.25, onde 25% das amostras são dedicadas ao treino e 75% das amostras são utilizadas para teste
etreino, eteste, streino, steste = train_test_split(entradas, saidas, test_size = 0.25)

knn.fit(etreino, streino)
previsor = knn.predict(eteste)

margem_acertos = metrics.accuracy_score(steste, previsor)

#cria um loop para ir mudando a distancia dos vi
valores_k = {}
k=1
while k < 25:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(etreino, streino)
    previsores_k = knn.predict(eteste)
    acertos = metrics.accuracy_score(steste, previsores_k)
    valores_k[k] = acertos
    k += 1

# Cria gráficos dos meus valores
# Recebe os valores em forma de lista e chave:valor de valores_k
plt.plot(list(valores_k.keys()),
         list(valores_k.values()))

#Define os rótulos do gráfico
plt.xlabel('Valores de K')
plt.ylabel('Performance')

regl = LogisticRegression()
regl.fit(etreino, streino)

#mostra em qual valor ele se encaixa, no caso 2 - viginica
print(regl.predict([[6.2,3.4,5.4,2.3]]))
#Mostra a probabilidade de cada saída.
print(regl.predict_proba([[6.2,3.4,5.4,2.3]]))

previsor_regl = regl.predict(eteste)
margem_acertos_regl = metrics.accuracy_score(steste, previsor_regl)