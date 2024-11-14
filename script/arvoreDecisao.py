import joblib
import pandas as pd
model = joblib.load('mariaEduarda.pkl')

# Previsão para um novo exemplo
sepal_length = float(input("Digite o valor da Sepal Length (cm): "))
sepal_width = float(input("Digite o valor da Sepal Width (cm): "))
petal_length = float(input("Digite o valor da Petal Length (cm): "))

feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
target_names = ['setosa', 'versicolor', 'virginica']

exemplo = pd.DataFrame([[sepal_length, sepal_width, petal_length]], columns=feature_names)
predicted = model.predict(exemplo)
print(f"Predição: {target_names[predicted[0]]}")

joblib.dump(model, 'mariaEduarda.pkl')