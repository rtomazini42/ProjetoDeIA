### Sobre a classificação de sentimentos

Fizemos uma base de treinamento manual com sentimento positivo e negativo com relação a cada Tweet em uma classificação binaria, depois colocamos o NLTK para classificar também e nos dar o feedback dele, com base nisso colocamos diferentes algoritmos para classificar a mesma dataset e comparamos os resultados para ver a taxa de acerto.

Etapa de pré processamento:
Após a coleta de dados no dia 24/03/2023 as 6:00 da manhã dos seguintes termos na plataforma twitter: "CLT","Recife", "Camaragibe","Olinda", "UFRPE", "UFPE", "@prefrecife","APAC", "@SantaCruzFC", "@sportrecife", "@nauticope", "@recifeordinario","@nautiluslink" e "@mimimidias". Fizemos uma checagem manual humana de positivo, negativo ou neutro.
O script de coleta é o scriptWS.py.

Após isso juntei todos em um único arquivo antes de passar os processamentos automaticos via o script juntador.py na pasta "DadosScrapados", resultando no arquivoResultado.csv.

Passei o pré processamento em NLTK resultando no arquivoResultadoAtualizado.csv.

