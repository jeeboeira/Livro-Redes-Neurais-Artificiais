import numpy as numpy
#Importa os modelos prontos de Json
from keras.models import model_from_json

arquivo = open(r'C:\Users\ti2\TestVsCode\Livro-Redes-Neurais-Artificiais-1\Redes_Neurais_Artificiais\Arquivos\Oito\classificador_binario.json', 'r')
# r no início de raw |r no fim de read
estrutura_rede = arquivo.read()
arquivo.close()