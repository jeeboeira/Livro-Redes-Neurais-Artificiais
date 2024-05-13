import numpy as np

#IMPLEMENTAÇÃO DA TABELA AND, PARA APRENDIZADO

# Cria Arrays no numpy
entradas = np.array([[0,0], [0,1], [1,0], [1,1]])
saidas = np.array([0, 0, 0, 1])
pesos = np.array([0.0, 0.0])


taxaAprendizado = 0.1

# Retorna o produto escalar de e sobre p
# Faz cálculo de entradas e pesos com uso da biblioteca numpy
def Soma(e, p):
    return e.dot(p)

# Recebe o produto escalar de entradas sobre pesos
s = Soma(entradas, pesos)

# Cria uma Step Function
#Função para ativar ou não o neurônio
def stepFunction(soma):
    if ( soma >= 1):
        return 1
    return 0

# Função para comparar se valores estão corretos
def calculoSaida(reg):
    s = reg.dot(pesos)
    return stepFunction(s)

#Bloco aprendizado de máquina
def aprendeAtualiza():
    erroTotal = 1
    while (erroTotal != 0):
        erroTotal = 0
        for i in range (len(saidas)):
            calcSaida = calculoSaida(np.array(entradas[i]))
            erro = abs(saidas[i] - calcSaida) #abs cálculo de forma absoluta, não considera o sinal negativo
            erroTotal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizado * entradas[i][j] * erro)
                print('Pesos Atualizados> ' + str(pesos[j]))
        print('Total de Erros: ' +str(erroTotal))

aprendeAtualiza()