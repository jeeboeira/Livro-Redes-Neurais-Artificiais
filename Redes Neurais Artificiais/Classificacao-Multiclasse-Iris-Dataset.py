from sklearn.datasets import load_iris

base = load_iris()

print(base.data)
print(base.target)
print(base.target_names)

entradas = base.data
saidas = base.target
rotulos = base.target_names