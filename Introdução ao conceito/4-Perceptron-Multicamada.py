import numpy as np

#IMPLEMENTAÇÃO DA TABELA XOR, PARA APRENDIZADO DE MÁQUINA

# Cria Arrays no numpy
# Dados de entrada
entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])
# Pesos atribuídos aleatóriamente
pesos0 = np.array([[-0.424, -0.740, -0.961],
                   [0.358, -0.577, -0.469]])
pesos1 = np.array([[-0.017], [-0.893], [0.148]])
# Saída esperada
saidas = np.array([[0], [1], [1], [0]])

# Número de vezes que a rede será treinada
nTreinos = 10000
# Define uma curva de aprendizado
# Este valor pode também piorar a eficiência, ou entrar em loop
taxaAprendizado = 0.3
#Constante para correção da margem de erro
momentum = 1

#Função sigmoid, substituindo a função degrau
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma)) #exp é exponênciação

for i in range(nTreinos):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)

    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)

    # Variáveis criadas por convenção para avaliar a eficiência do processamento da rede
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))# mean transforma em porcentagem


    def sigmoidDerivada(sig):
        return sig * (1 - sig)
    sigDerivada = sigmoid(0.5)
    sigDerivada1 = sigmoidDerivada(sigDerivada)

    derivadaSaida = sigmoidDerivada(camadaSaida)
    # Parâmetro usado como referência para correto ajuste da descida do gradiente
    #realiza um ajuste fino dos pesos e minimiza suas margens de erro
    deltaSaida = erroCamadaSaida * derivadaSaida


    pesos1Transposta = pesos1.T #.T transposta o array ou matriz
    deltaSaidaXpesos = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXpesos * sigmoidDerivada(camadaOculta)

    # Backpropagation
    camadaOcultaTransposta = camadaOculta.T
    pesos3 = camadaOcultaTransposta.dot(deltaSaida) # Produto escalar de camadaOcultaTransposta pelo deltaSaida
    pesos1 = (pesos1 * momentum) + (pesos3 * taxaAprendizado)

    camadaEntradaTransposta = camadaEntrada.T
    pesos4 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momentum) + (pesos4 * taxaAprendizado)

    
    print(f"Treino: {i} margem de erro: " + str(mediaAbsoluta))
