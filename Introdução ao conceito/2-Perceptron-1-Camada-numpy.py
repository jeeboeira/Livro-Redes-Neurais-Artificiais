import numpy as np

#crio Array no numpy
entradas = np.array([1, 9, 5])
pesos = np.array([0.8, 0.1, 0])

#Retorna o produto escalar de e sobre p
#Faz cálculo de entradas e pesos com uso da biblioteca numpy
def Soma(e, p):
    return e.dot(p)

s = Soma(entradas, pesos)

def stepFunction(Soma):
    if ( s >= 1):
        return 1
    return 0

saida = stepFunction(s)

print(s)
print(saida)