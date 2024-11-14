import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn import tree
from joblib import dump

# Carregando o conjunto de dados
# dados = load_breast_cancer()
database = pd.read_csv("files/breast_cancer.csv")

# Definindo atributos e rótulos
nome_atributos = database.drop(columns=['classes']).columns
atributos = database[nome_atributos]
rotulos = database["classes"] 

X = atributos 
y = rotulos

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=13)

# Criar o modelo de árvore de decisão
modelo = DecisionTreeClassifier(max_depth=None, random_state=13)

# Modelo de árvore de decisão
modelo.fit(X_train, y_train)

# Previsões
y_pred = modelo.predict(X_test)


# Calculando a precisão, recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
# Avaliando o modelo
print(f'Acurácia do modelo:{accuracy:.2f}')
print(f'Precisão: {precision:.2f}')
print((f'Recall: {recall:.2f}'))
print((f'F1-Score: {f1:.2f}'))

print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))
print("\nMatriz de confusão:\n", confusion_matrix(y_test, y_pred))

# Visualizar a árvore de decisão
plt.figure(figsize=(10, 7))
tree.plot_tree(modelo, feature_names=X.columns, class_names=['Benigno', 'Maligno'], filled=True)
plt.title("Árvore de Decisão para Diagnóstico")
plt.show()

colors=['#35b2de','#ffcb5a']

labels=database['classes'].value_counts().index
plt.pie(database['classes'].value_counts(),autopct='%1.1f%%',colors=colors) 
plt.legend(labels,bbox_to_anchor=(1.25,1),) 
plt.title('Porcentagem: Benignos x Malignos ')
plt.show()


res = input("Deseja salvar o modelo? s/n: ")
if res == 's':
    dump(modelo, 'model/arvore.pkl')
    print("Modelo salvo com sucesso!")
else:
    sys.exit("Saída do programa")