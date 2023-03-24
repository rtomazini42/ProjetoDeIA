import pandas as pd

df = pd.read_csv('arquivoResultadoAtualizado.csv')

# Contar frequências de sentimento e sentimentoNLTK
ct = pd.crosstab(df.sentimento, df.sentimentoNLTK, normalize='index')

# Calcular porcentagens de acerto para cada categoria
acertos_pos = ct.loc['pos', 'pos']
acertos_neg = ct.loc['neg', 'neg']
acertos_neu = ct.loc['neu', 'neu']

# Imprimir resultados
print(f"Porcentagem de acertos para sentimento positivo: {acertos_pos:.2%}")
print(f"Porcentagem de acertos para sentimento negativo: {acertos_neg:.2%}")
print(f"Porcentagem de acertos para sentimento neutro: {acertos_neu:.2%}")
