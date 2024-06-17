import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
#Importa Suport Vector Machine, altamente usada na classificação de imagens
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
#Carrega funções de manipulação de imagem
import matplotlib.image as mimg
# Função para redimensionar imagens
from skimage.transform import resize
# Realiza a regressão logística
from sklearn.linear_model import LogisticRegression

base = datasets.load_digits()
entradas = base.data
saidas = base.target

#print(entradas[0])
#print(base.images[0])

#Printa uma imagem
plt.figure(figsize = (2, 2))
plt.imshow(base.images[0],
           cmap = plt.cm.gray_r)


etreino, eteste, streino, steste = train_test_split(entradas,
                                                    saidas,
                                                    test_size = 0.1,
                                                    random_state = 2) # Método de separação das amostras randomico
                                                    #não pegando dados com proximidade igual o u menor a 2 números

classificador = svm.SVC() # Support Vector Classification
classificador.fit(etreino, streino)
previsor = classificador.predict(eteste)

margem_acerto = metrics.accuracy_score(steste, previsor)

# Carrega uma imagem como array nump
imagem = mimg.imread(r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_convolucionais\Arquivos_Convolucionais\1Um\num2.1.png')

#Função para tratar a imagem
def rgb2gray(imagem):
    # Redimensiona a imagem para 8x8 pixels (outra forma de reduzir para 64 posições)
    imagem = resize(imagem, (8, 8), anti_aliasing = True)
    # Seleciona os canais RGB da imagem
    rgb_channels = imagem[...,:3]
     # Aplica os pesos de conversão para obter a escala de cinza
    imagem = np.dot(rgb_channels, [0.299, 0.587, 0.114])
    # Ajusta o intervalo da escala de cinza para 0-16
    imagem = (16 - (imagem * 16)).astype(int)
    # Aplica a função flatten que converte uma array numpy multidimensional, e um array unidimensional
    return imagem.flatten()

#Utilizo Suport Vector Machine para fazer a previsão da imagem que eu fornecer
identificador = svm.SVC()
identificador.fit(entradas, saidas)
previsor_id = identificador.predict([rgb2gray(imagem)])
print(previsor_id)

#Faz uma regressão logistica
logr = LogisticRegression()
logr.fit(etreino, streino)
previsor_logr = logr.predict(eteste)
acerto_logr = metrics.accuracy_score(steste, previsor_logr)
print(acerto_logr)

#Regressor
regressor = LogisticRegression()
regressor.fit(entradas, saidas)
previsor_regl = regressor.predict([rgb2gray(imagem)])
print(previsor_regl)