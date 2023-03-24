import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analisar_sentimento(texto):
    sia = SentimentIntensityAnalyzer()
    polaridade = sia.polarity_scores(texto)['compound']
    if polaridade > 0:
        return 'pos'
    elif polaridade < 0:
        return 'neg'
    else:
        return 'neu'

df = pd.read_csv('arquivoResultado.csv')
df['sentimentoNLTK'] = ''

for index, row in df.iterrows():
    sentimento = analisar_sentimento(row['texto'])
    df.at[index, 'sentimentoNLTK'] = sentimento

df.to_csv('arquivoResultadoAtualizado.csv', index=False)
