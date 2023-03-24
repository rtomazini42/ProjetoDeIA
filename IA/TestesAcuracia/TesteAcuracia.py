
import time #para os calculos de tempo

start_time = 0
end_time = 0

def comeca_contagem():
    start_time = time.time()


def termina_contagem():
    end_time = time.time()
    total_time = end_time - start_time

    # exibe o tempo total de execução
    print("####################################################")
    print(f"#Tempo total de execução: {total_time} segundos#")
    print("####################################################")

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import f1_score



print("+++NaiveBayes+++")
#NaiveBayes testando na curadoria humana
print("NaiveBayes testando na curadoria humana")
# Leitura do arquivo
df = pd.read_csv('arquivoResultadoAtualizado.csv')

# Pré-processamento dos dados
comeca_contagem()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['texto'].values.astype('U'))
y = df['sentimento'].values

# Definição do modelo
modelo = MultinomialNB()

# Definição dos folds para a validação cruzada
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Definição do número de repetições da validação cruzada
num_repeticoes = 5

# Execução da validação cruzada para o cálculo da acurácia e do F1-score
for i in range(num_repeticoes):
    resultados = cross_validate(modelo, X, y, cv=kfold, n_jobs=-1, return_train_score=False, scoring=['accuracy'])

    # Cálculo da média dos resultados de acurácia e do F1-score
    acuracia_media = resultados['test_accuracy'].mean()
#   f1_score_medio = resultados['test_f1'].mean()

    print(f"Acurácia média da {i+1}ª repetição: {acuracia_media}")
#    print(f"F1-score médio da {i+1}ª repetição: {f1_score_medio}")
termina_contagem()

print("--------------------------------------")

#NaiveBayes testando na classificação NLTK
print("NaiveBayes testando na classificação NLTK")


# Leitura do arquivo
df = pd.read_csv('arquivoResultadoAtualizado.csv')

# Pré-processamento dos dados
comeca_contagem()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['texto'].values.astype('U'))
y = df['sentimentoNLTK'].values

# Definição do modelo
modelo = MultinomialNB()

# Definição dos folds para a validação cruzada
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Definição do número de repetições da validação cruzada
num_repeticoes = 5

# Execução da validação cruzada
for i in range(num_repeticoes):
    resultados = cross_validate(modelo, X, y, cv=kfold, n_jobs=-1, return_train_score=False, scoring=['accuracy'])

    # Cálculo da média dos resultados de acurácia e do F1-score
    acuracia_media = resultados['test_accuracy'].mean()
#    f1_score_medio = resultados['test_f1'].mean()

    print(f"Acurácia média da {i+1}ª repetição: {acuracia_media}")
#    print(f"F1-score médio da {i+1}ª repetição: {f1_score_medio}")
termina_contagem()

print("--------------------------------------")
print("+++KNN+++")
print("--------------------------------------")

#KNN testando na curadoria humana con KNN = 3
print("KNN testando na curadoria humana com KNN = 3")


# Leitura do arquivo
df = pd.read_csv('arquivoResultadoAtualizado.csv')

# Pré-processamento dos dados
comeca_contagem()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['texto'].values.astype('U'))
y = df['sentimento'].values

# Definição do modelo knn = 5
modelo = KNeighborsClassifier(n_neighbors=3)

# Definição dos folds para a validação cruzada
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Definição do número de repetições da validação cruzada
num_repeticoes = 5

# Execução da validação cruzada
for i in range(num_repeticoes):
    resultados = cross_validate(modelo, X, y, cv=kfold, n_jobs=-1, return_train_score=False, scoring=['accuracy'])

    # Cálculo da média dos resultados de acurácia e do F1-score
    acuracia_media = resultados['test_accuracy'].mean()
#    f1_score_medio = resultados['test_f1'].mean()

    print(f"Acurácia média da {i+1}ª repetição: {acuracia_media}")
#    print(f"F1-score médio da {i+1}ª repetição: {f1_score_medio}")
termina_contagem()


print("--------------------------------------")

#KNN testando na curadoria humana con KNN = 5
print("KNN testando na curadoria humana com KNN = 5")


# Leitura do arquivo
df = pd.read_csv('arquivoResultadoAtualizado.csv')

# Pré-processamento dos dados
comeca_contagem()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['texto'].values.astype('U'))
y = df['sentimento'].values

# Definição do modelo knn = 5
modelo = KNeighborsClassifier(n_neighbors=5)

# Definição dos folds para a validação cruzada
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Definição do número de repetições da validação cruzada
num_repeticoes = 5

# Execução da validação cruzada
for i in range(num_repeticoes):
    resultados = cross_validate(modelo, X, y, cv=kfold, n_jobs=-1, return_train_score=False, scoring=['accuracy'])

    # Cálculo da média dos resultados de acurácia e do F1-score
    acuracia_media = resultados['test_accuracy'].mean()
#    f1_score_medio = resultados['test_f1'].mean()

    print(f"Acurácia média da {i+1}ª repetição: {acuracia_media}")
#    print(f"F1-score médio da {i+1}ª repetição: {f1_score_medio}")

termina_contagem()

print("--------------------------------------")

#KNN testando na curadoria humana con KNN = 7
print("KNN testando na curadoria humana com KNN = 7")


# Leitura do arquivo
df = pd.read_csv('arquivoResultadoAtualizado.csv')

# Pré-processamento dos dados
comeca_contagem()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['texto'].values.astype('U'))
y = df['sentimento'].values

# Definição do modelo knn = 5
modelo = KNeighborsClassifier(n_neighbors=7)

# Definição dos folds para a validação cruzada
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Definição do número de repetições da validação cruzada
num_repeticoes = 5

# Execução da validação cruzada
for i in range(num_repeticoes):
    resultados = cross_validate(modelo, X, y, cv=kfold, n_jobs=-1, return_train_score=False, scoring=['accuracy'])

    # Cálculo da média dos resultados de acurácia e do F1-score
    acuracia_media = resultados['test_accuracy'].mean()
#    f1_score_medio = resultados['test_f1'].mean()

    print(f"Acurácia média da {i+1}ª repetição: {acuracia_media}")
#    print(f"F1-score médio da {i+1}ª repetição: {f1_score_medio}")
termina_contagem()

print("--------------------------------------")

print("--------------------------------------")

#KNN testando na NLTK con KNN = 3
print("KNN testando na NLTK com KNN = 3")


# Leitura do arquivo
df = pd.read_csv('arquivoResultadoAtualizado.csv')

# Pré-processamento dos dados
comeca_contagem()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['texto'].values.astype('U'))
y = df['sentimentoNLTK'].values

# Definição do modelo knn = 5
modelo = KNeighborsClassifier(n_neighbors=3)

# Definição dos folds para a validação cruzada
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Definição do número de repetições da validação cruzada
num_repeticoes = 5

# Execução da validação cruzada
for i in range(num_repeticoes):
    resultados = cross_validate(modelo, X, y, cv=kfold, n_jobs=-1, return_train_score=False, scoring=['accuracy'])

    # Cálculo da média dos resultados de acurácia e do F1-score
    acuracia_media = resultados['test_accuracy'].mean()
#    f1_score_medio = resultados['test_f1'].mean()

    print(f"Acurácia média da {i+1}ª repetição: {acuracia_media}")
#    print(f"F1-score médio da {i+1}ª repetição: {f1_score_medio}")
termina_contagem()


print("--------------------------------------")

#KNN testando na NLTK KNN = 5
print("KNN testando na NLTK com KNN = 5")


# Leitura do arquivo
df = pd.read_csv('arquivoResultadoAtualizado.csv')

# Pré-processamento dos dados
comeca_contagem()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['texto'].values.astype('U'))
y = df['sentimentoNLTK'].values

# Definição do modelo knn = 5
modelo = KNeighborsClassifier(n_neighbors=5)

# Definição dos folds para a validação cruzada
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Definição do número de repetições da validação cruzada
num_repeticoes = 5

# Execução da validação cruzada
for i in range(num_repeticoes):
    resultados = cross_validate(modelo, X, y, cv=kfold, n_jobs=-1, return_train_score=False, scoring=['accuracy'])

    # Cálculo da média dos resultados de acurácia e do F1-score
    acuracia_media = resultados['test_accuracy'].mean()
#    f1_score_medio = resultados['test_f1'].mean()

    print(f"Acurácia média da {i+1}ª repetição: {acuracia_media}")
#    print(f"F1-score médio da {i+1}ª repetição: {f1_score_medio}")
termina_contagem()

print("--------------------------------------")

#KNN testando  na NLTK  con KNN = 7
print("KNN testando na NLTK com KNN = 7")


# Leitura do arquivo
df = pd.read_csv('arquivoResultadoAtualizado.csv')

# Pré-processamento dos dados
comeca_contagem()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['texto'].values.astype('U'))
y = df['sentimentoNLTK'].values

# Definição do modelo knn = 5
modelo = KNeighborsClassifier(n_neighbors=7)

# Definição dos folds para a validação cruzada
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Definição do número de repetições da validação cruzada
num_repeticoes = 5

# Execução da validação cruzada
for i in range(num_repeticoes):
    resultados = cross_validate(modelo, X, y, cv=kfold, n_jobs=-1, return_train_score=False, scoring=['accuracy'])

    # Cálculo da média dos resultados de acurácia e do F1-score
    acuracia_media = resultados['test_accuracy'].mean()
#    f1_score_medio = resultados['test_f1'].mean()

    print(f"Acurácia média da {i+1}ª repetição: {acuracia_media}")
#    print(f"F1-score médio da {i+1}ª repetição: {f1_score_medio}")
termina_contagem()

print("--------------------------------------")
print("Árvore de decisão")
print("--------------------------------------")

#Árvore de decisão para curadoria humana
print("Árvore de decisão para curadoria humana")

# Leitura do arquivo
df = pd.read_csv('arquivoResultadoAtualizado.csv')

# Pré-processamento dos dados
comeca_contagem()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['texto'].values.astype('U'))
y = df['sentimento'].values

# Definição do modelo
modelo = DecisionTreeClassifier()

# Definição dos folds para a validação cruzada
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Definição do número de repetições da validação cruzada
num_repeticoes = 5

# Execução da validação cruzada
for i in range(num_repeticoes):
    resultados = cross_validate(modelo, X, y, cv=kfold, n_jobs=-1, return_train_score=False, scoring=['accuracy'])

    # Cálculo da média dos resultados de acurácia e do F1-score
    acuracia_media = resultados['test_accuracy'].mean()
#    f1_score_medio = resultados['test_f1'].mean()

    print(f"Acurácia média da {i+1}ª repetição: {acuracia_media}")
#    print(f"F1-score médio da {i+1}ª repetição: {f1_score_medio}")
termina_contagem()


print("--------------------------------------")
#Árvore de decisão para curadoria humana
print("Árvore de decisão para NLTK")

# Leitura do arquivo
df = pd.read_csv('arquivoResultadoAtualizado.csv')

# Pré-processamento dos dados
comeca_contagem()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['texto'].values.astype('U'))
y = df['sentimentoNLTK'].values

# Definição do modelo
modelo = DecisionTreeClassifier()

# Definição dos folds para a validação cruzada
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Definição do número de repetições da validação cruzada
num_repeticoes = 5

# Execução da validação cruzada
for i in range(num_repeticoes):
    resultados = cross_validate(modelo, X, y, cv=kfold, n_jobs=-1, return_train_score=False, scoring=['accuracy'])

    # Cálculo da média dos resultados de acurácia e do F1-score
    acuracia_media = resultados['test_accuracy'].mean()
#    f1_score_medio = resultados['test_f1'].mean()

    print(f"Acurácia média da {i+1}ª repetição: {acuracia_media}")
#    print(f"F1-score médio da {i+1}ª repetição: {f1_score_medio}")
termina_contagem()