# Perceptron de uma camada

entradas = [1, 9, 5]
pesos = [0.8, 0.1, 0]

#Faz cálculo de entradas e pesos
def soma(e, p):
  s = 0
  for i in range(3):
    s += e[i] * p[i]
  return s
  print (s)


s = soma(entradas, pesos)

print(s)

#Função para ativar ou não o neurônio
def stepFunction(s):
  if (s >= 1):
    return 1
  return 0

saida = stepFunction(s)

print(saida)