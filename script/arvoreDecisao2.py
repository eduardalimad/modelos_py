import joblib
import pandas as pd
modelo = joblib.load('model/arvore.pkl')

# Previsão para um novo exemplo
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", 
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", 
    "mean fractal dimension", "radius error", "texture error", "perimeter error", 
    "area error", "smoothness error", "compactness error", "concavity error", 
    "concave points error", "symmetry error", "fractal dimension error", "worst radius", 
    "worst texture", "worst perimeter", "worst area", "worst smoothness", 
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", 
    "worst fractal dimension"
]

exemplo_01 = [[
    18.25, 11.20, 130.5, 1043.2, 1.220, 2.801, 3.108, 1.506, 2.510, 7.902,
    1.103, 9.210, 8.795, 156.7, 6.512, 4.982, 5.412, 1.605, 3.101, 6.245,
    25.96, 18.02, 190.3, 2105.5, 1.650, 6.700, 7.203, 2.682, 4.715, 1.205
]]
exemplo_02 = [[
    20.10, 13.50, 140.3, 1200.5, 1.350, 3.050, 3.200, 1.550, 2.800, 8.500,
    1.200, 10.500, 9.300, 170.2, 6.800, 5.100, 5.800, 1.700, 3.200, 6.500,
    27.10, 19.20, 200.1, 2250.3, 1.700, 7.000, 7.500, 2.800, 5.000, 1.300
]]

exemplo_03 = [[
    17.80, 10.90, 125.7, 990.1, 1.180, 2.700, 3.000, 1.400, 2.400, 7.500,
    1.000, 8.900, 8.600, 150.5, 6.300, 4.800, 5.200, 1.500, 3.000, 6.000,
    24.80, 17.50, 180.7, 2000.4, 1.600, 6.500, 7.000, 2.600, 4.600, 1.100
]]


# Cria o DataFrame com os nomes das colunas corretos
exemplo = pd.DataFrame(exemplo_01, columns=feature_names)

# Faz a previsão
predicted = modelo.predict(exemplo)
print(f"Predição: {predicted[0]}")

