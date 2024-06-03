# Repositório dos dados breast cancer
from ucimlrepo import fetch_ucirepo
import numpy as np

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

entradas = breast_cancer_wisconsin_diagnostic.data.features

X = entradas.iloc[68].to_numpy()
print(X)
