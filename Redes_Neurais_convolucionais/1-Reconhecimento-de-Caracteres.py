import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
#Importa Suport Vector Machine, altamente usada na classificação de imagens
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.image as mimg

base = datasets.load_digits()
entradas = base.data
saidas = base.target

print(entradas[0])
print(base.images[0])

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

imagem = mimg.imread(r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_convolucionais\Arquivos_Convolucionais\1Um\num2.png')
print(imagem)

def rgb2gray(rgb):
    # Recebe o produto escalar de todas as linhas e das 3 colunas vistas anteriormente
        #sobre os valores 0.299,0.587,0.114, uma convenção de conversão rgb
    img_array = np.dot(rgb[...,:3],[0.299,0.587,0.114])
    # Trsanforma o resultado para inteiro com a função .astype
    img_array2 = (16 - (img_array * 16)).astype(int)
    # Converte para uma matriz indexada de 64 valores, em um intervalo de 0 a 16
        #escala de cinza 
    img_array3 = img_array2.flatten()
    return img_array3

rgb2gray(imagem)