import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate, KFold
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import f1_score

print("+++NaiveBayes+++")
print("NaiveBayes testando na curadoria humana")
# Leitura do arquivo
df = pd.read_csv('arquivoResultadoAtualizado.csv')

# Pré-processamento dos dados
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
   

    print(f"Acurácia média da {i+1}ª repetição: {acuracia_media}")
   


#f1_score_medio = resultados['test_f1'].mean()
#print(f"F1-score médio da {i+1}ª repetição: {f1_score_medio}")
