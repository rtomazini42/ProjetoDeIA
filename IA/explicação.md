### Sobre a classificação de sentimentos

Fizemos uma base de treinamento manual com sentimento positivo e negativo com relação a cada Tweet em uma classificação binaria, depois colocamos o NLTK para classificar também e nos dar o feedback dele, com base nisso colocamos diferentes algoritmos para classificar a mesma dataset e comparamos os resultados para ver a taxa de acerto.

#### Etapa de pré processamento:
Após a coleta de dados no dia 24/03/2023 as 6:00 da manhã dos seguintes termos na plataforma twitter: "CLT","Recife", "Camaragibe","Olinda", "UFRPE", "UFPE", "@prefrecife","APAC", "@SantaCruzFC", "@sportrecife", "@nauticope", "@recifeordinario","@nautiluslink" e "@mimimidias". Fizemos uma checagem manual humana de positivo, negativo ou neutro.
O script de coleta é o scriptWS.py.

Após isso juntei todos em um único arquivo antes de passar os processamentos automaticos via o script juntador.py na pasta "DadosScrapados", resultando no arquivoResultado.csv.

Passei o pré processamento em NLTK resultando no arquivoResultadoAtualizado.csv.

Usando do comparativoNLTK.py obtivemos os seguintes indices:
Porcentagem de acertos para sentimento positivo: 2.15%
Porcentagem de acertos para sentimento negativo: 16.07%
Porcentagem de acertos para sentimento neutro: 79.82%

Resultando que o NLTK está apresentando indice de erros muito grandes em comparação á interpretação humana nesse conjunto especifico de tweets, uma das explicações se deve ao fato do uso de muita linguagem regionalista nesse conjunto de dados, de forma proposital escolhemos esse conjunto. 

#### Comparações e exploração dos dados


Os experimentos devem ser executados de acordo com o esquema abaixo para cada uma das bases de dados geradas (tanto a base de dados “brutos” quanto a base de dados pré-processadas):
* Deve-se executar o 10-fold cross-validation 5 vezes para cada base de dados, com cada uma das cinco execuções partindo de uma distribuição aleatória dos dados entre cada fold, resultando em um total de 50 experimentos por base de dados (10 x 5).
* Em cada um dos 50 experimentos, os conjuntos de treinamento e teste devem ser mantido o mesmo para cada algoritmo a ser testado (mesmo ponto de partida para cada modelo), de modo a obter-se uma avaliação justa dos resultados.
* Três algoritmos devem ser testados e comparados: 
Naïve Bayes;
K-Vizinhos Mais Próximos (K-NN) - variando-se 3 vezes o número do parâmetro k;
Árvore de decisão 

* Ao menos duas métricas (índices) de avaliação deverão ser empregadas na análise experimental, além do tempo médio de execução de cada um dos algoritmos. 
* As métricas escolhidas devem ser justificadas pela Revisão da Literatura.


##### Para a análise de sentimentos de tweets, usamos as seguintes métricas de avaliação:

    Acurácia (Accuracy): é a métrica mais comum para avaliar algoritmos de classificação e consiste na proporção de predições corretas em relação ao total de predições realizadas. É calculada pela fórmula:

    acurácia = (verdadeiros positivos + verdadeiros negativos) / total de predições

    Essa métrica é amplamente utilizada em trabalhos de análise de sentimentos, como por exemplo em (1) e em (2).

    F1-score: essa métrica leva em consideração a precisão e a revocação (recall) do algoritmo e é útil quando há um desbalanceamento na distribuição das classes no conjunto de dados. É calculada pela fórmula:

    F1-score = 2 * (precisão * recall) / (precisão + recall)

    Essa métrica é recomendada em trabalhos que possuem classes desbalanceadas, como é o caso da análise de sentimentos em tweets (4).

1 - Yadav, Ashima, and Dinesh Kumar Vishwakarma. "Sentiment analysis using deep learning architectures: a review." Artificial Intelligence Review 53.6 (2020): 4335-4385.
2 - Kathuria, Ramandeep Singh, et al. "Real time sentiment analysis on twitter data using deep learning (Keras)." 2019 international conference on computing, communication, and intelligent systems (ICCCIS). IEEE, 2019.
3 - Alonso, Miguel A., et al. "Sentiment analysis for fake news detection." Electronics 10.11 (2021): 1348.
4 - Dang, N.C.; Moreno-García, M.N.; De la Prieta, F. Sentiment Analysis Based on Deep Learning: A Comparative Study. Electronics 2020, 9, 483. https://doi.org/10.3390/electronics9030483 


## Sobre o TesteAcuracia.py presentre em IA/TestesAcuracia
Esse código é um script em Python que realiza classificações usando Naive Bayes e K-Nearest Neighbors (KNN) em um conjunto de dados de textos. Ele usa a biblioteca Scikit-Learn para criar modelos de classificação e validação cruzada para avaliar a acurácia dos modelos.

O script consiste em duas partes: NaiveBayes testando na curadoria humana e NaiveBayes testando na classificação NLTK, seguidas por KNN testando na curadoria humana com diferentes valores de K.

Na primeira parte, o script usa o algoritmo Naive Bayes para classificar o sentimento de textos em um conjunto de dados. Ele lê um arquivo CSV contendo os textos e seus respectivos sentimentos, aplica a técnica de vetorização CountVectorizer para converter os textos em uma representação numérica e, em seguida, usa o Naive Bayes para criar um modelo de classificação. Em seguida, ele executa a validação cruzada para avaliar o desempenho do modelo em 10 partes do conjunto de dados, repetindo o processo 5 vezes e calculando a acurácia média em cada repetição.

Na segunda parte, o script repete o processo anterior, mas em vez de usar o sentimento humano como variável de destino, ele usa a classificação do NLTK (Natural Language Toolkit), uma biblioteca de processamento de linguagem natural para Python.

Na terceira parte, o script usa o algoritmo KNN para classificar os textos em um conjunto de dados. Ele repete o processo de leitura do arquivo CSV e vetorização CountVectorizer. Em seguida, usa o KNN para criar um modelo de classificação com diferentes valores de K (3, 5 e 7) e executa a validação cruzada para avaliar a acurácia média em cada repetição.
Abaixo temos os resultados obtidos, juntamente com uma contagem de tempo de execução.


## Resultados dos testes de acuracia e tempo:
+++NaiveBayes+++
NaiveBayes testando na curadoria humana
Acurácia média da 1ª repetição: 0.6660210669384982
Acurácia média da 2ª repetição: 0.6660210669384982
Acurácia média da 3ª repetição: 0.6660210669384982
Acurácia média da 4ª repetição: 0.6660210669384982
Acurácia média da 5ª repetição: 0.6660210669384982
####################################################
#Tempo total de execução: 1679672086.7249763 segundos#
####################################################
--------------------------------------
NaiveBayes testando na classificação NLTK
Acurácia média da 1ª repetição: 0.88065749235474
Acurácia média da 2ª repetição: 0.88065749235474
Acurácia média da 3ª repetição: 0.88065749235474
Acurácia média da 4ª repetição: 0.88065749235474
Acurácia média da 5ª repetição: 0.88065749235474
####################################################
#Tempo total de execução: 1679672086.8631966 segundos#
####################################################
--------------------------------------
+++KNN+++
--------------------------------------
KNN testando na curadoria humana com KNN = 3
Acurácia média da 1ª repetição: 0.6040519877675842
Acurácia média da 2ª repetição: 0.6040519877675842
Acurácia média da 3ª repetição: 0.6040519877675842
Acurácia média da 4ª repetição: 0.6040519877675842
Acurácia média da 5ª repetição: 0.6040519877675842
####################################################
#Tempo total de execução: 1679672087.1218078 segundos#
####################################################
--------------------------------------
KNN testando na curadoria humana com KNN = 5
Acurácia média da 1ª repetição: 0.6012742099898063
Acurácia média da 2ª repetição: 0.6012742099898063
Acurácia média da 3ª repetição: 0.6012742099898063
Acurácia média da 4ª repetição: 0.6012742099898063
Acurácia média da 5ª repetição: 0.6012742099898063
####################################################
#Tempo total de execução: 1679672087.352609 segundos#
####################################################
--------------------------------------
KNN testando na curadoria humana com KNN = 7
Acurácia média da 1ª repetição: 0.5939007815154604
Acurácia média da 2ª repetição: 0.5939007815154604
Acurácia média da 3ª repetição: 0.5939007815154604
Acurácia média da 4ª repetição: 0.5939007815154604
Acurácia média da 5ª repetição: 0.5939007815154604
####################################################
#Tempo total de execução: 1679672087.601194 segundos#
####################################################
--------------------------------------
--------------------------------------
KNN testando na NLTK com KNN = 3
Acurácia média da 1ª repetição: 0.8889823309548082
Acurácia média da 2ª repetição: 0.8889823309548082
Acurácia média da 3ª repetição: 0.8889823309548082
Acurácia média da 4ª repetição: 0.8889823309548082
Acurácia média da 5ª repetição: 0.8889823309548082
####################################################
#Tempo total de execução: 1679672087.8299298 segundos#
####################################################
--------------------------------------
KNN testando na NLTK com KNN = 5
Acurácia média da 1ª repetição: 0.8778797145769621
Acurácia média da 2ª repetição: 0.8778797145769621
Acurácia média da 3ª repetição: 0.8778797145769621
Acurácia média da 4ª repetição: 0.8778797145769621
Acurácia média da 5ª repetição: 0.8778797145769621
####################################################
#Tempo total de execução: 1679672088.0604086 segundos#
####################################################
--------------------------------------
KNN testando na NLTK com KNN = 7
Acurácia média da 1ª repetição: 0.8686289500509684
Acurácia média da 2ª repetição: 0.8686289500509684
Acurácia média da 3ª repetição: 0.8686289500509684
Acurácia média da 4ª repetição: 0.8686289500509684
Acurácia média da 5ª repetição: 0.8686289500509684
####################################################
#Tempo total de execução: 1679672088.2850685 segundos#
####################################################
--------------------------------------
Árvore de decisão
--------------------------------------
Árvore de decisão para curadoria humana
Acurácia média da 1ª repetição: 0.5762487257900102
Acurácia média da 2ª repetição: 0.5929323819232076
Acurácia média da 3ª repetição: 0.5957016649677199
Acurácia média da 4ª repetição: 0.590137614678899
Acurácia média da 5ª repetição: 0.599413863404689
####################################################
#Tempo total de execução: 1679672089.245056 segundos#
####################################################
--------------------------------------
Árvore de decisão para NLTK
Acurácia média da 1ª repetição: 0.9722392116887528
Acurácia média da 2ª repetição: 0.9768688413183824
Acurácia média da 3ª repetição: 0.9759429153924566
Acurácia média da 4ª repetição: 0.9694614339109752
Acurácia média da 5ª repetição: 0.9759429153924566
####################################################
#Tempo total de execução: 1679672089.7561336 segundos#
####################################################

