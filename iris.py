# Importar bibliotecas
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar um classificador SVM
svm_classifier = SVC()

# Treinar o modelo
svm_classifier.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = svm_classifier.predict(X_test)

# Calcular a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'A precisão do modelo SVM é: {accuracy * 100:.2f}%')
