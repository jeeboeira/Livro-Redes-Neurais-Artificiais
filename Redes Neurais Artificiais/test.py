from ucimlrepo import fetch_ucirepo


breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
entradas = breast_cancer_wisconsin_diagnostic.data.features 
saidas = breast_cancer_wisconsin_diagnostic.data.targets

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

saidasEnc = lb.fit_transform(saidas)
print(saidasEnc)