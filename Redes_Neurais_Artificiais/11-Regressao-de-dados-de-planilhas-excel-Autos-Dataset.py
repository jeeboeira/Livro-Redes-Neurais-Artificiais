import pandas as pd

#importado o dataset, com encoding padrão
base = pd.read_csv(r"C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_Artificiais\Arquivos\Onze\autos.csv", encoding = "ISO-8859-1")

#Variável para eu acompanhar as mudanças
abaseOriginal = base

# Imprime as 5 primeiras linhas
print(base.head())

#Exclui as colunas pelo nome
base = base.drop(['index', 'dateCrawled', 'dateCreated', 'nrOfPictures', 'postalCode', 'lastSeen'], axis = 1)

#Aqui eu vejo o número de ocorrência de cada valor, como existem muitas inconsistencias, melhor dropar
print(base['name'].value_counts())
base = base.drop(['name', 'seller', 'offerType'], axis = 1)

#Aqui filtro amostras com valores absurdamente baixos
varTeste1 = base.loc[base.price <=10]
#Assim faço o método ao contrário, e salvo na mesma variável
base = base[base.price > 10]

#Aqui filtro os valores muito acima
varTeste2 = base.loc[base.price > 350000]
#Assim faço o método ao contrário, e salvo na mesma variável
base = base[base.price < 350000]