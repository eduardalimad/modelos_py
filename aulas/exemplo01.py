import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
from joblib import dump

# Carregando o conjunto de dados
conjunto_dados = pd.read_csv("./files/iris_small.csv", index_col=0)
df = pd.DataFrame(conjunto_dados)

print("Primeiro Dataset- flor de íris")

# Definindo atributos e rótulos
nome_atributos = conjunto_dados.columns[:3]
atributos = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']]
rotulos = df['classes']

# Exibindo informações sobre o dataset
print("Exemplos:", atributos.shape[0])
print("Atributos:", atributos.shape[1])
print("Classes distintas: ", rotulos.unique())

# Definindo X e y
X = atributos 
y = rotulos

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
print('Dimensões dos subconjuntos', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Modelo de árvore de decisão
model = DecisionTreeClassifier(max_depth=None, random_state=0)
model.fit(X_train, y_train)

# Matriz de confusão dos resultados
predicted = model.predict(X_train)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_train, predicted)).plot()

plt.grid(False)
plt.show()

# Calculando a acurácia
predicted = model.predict(X_test)

# Verificando se as dimensões são consistentes
if len(predicted) == len(y_test):
    accuracy = accuracy_score(y_test, predicted)
    print(f'Acurácia: {accuracy:.2f}')
else:
    print("Erro: As dimensões de 'predicted' e 'y_test' não coincidem.")

# Plotando a árvore de decisão
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
target_names = ['setosa', 'versicolor', 'virginica']

fig = plt.figure(figsize=(10, 7))
tree.plot_tree(model, feature_names=feature_names, class_names=target_names, filled=True)
plt.show()

# Previsão para um novo exemplo
sepal_length = float(input("Digite o valor da Sepal Length (cm): "))
sepal_width = float(input("Digite o valor da Sepal Width (cm): "))
petal_length = float(input("Digite o valor da Petal Length (cm): "))

exemplo = pd.DataFrame([[sepal_length, sepal_width, petal_length]], columns=feature_names)
predicted = model.predict(exemplo)
print(f"Predição: {target_names[predicted[0]]}")

dump(model, 'model/mariaEduarda.pkl')
