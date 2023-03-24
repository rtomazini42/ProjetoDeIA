import pandas as pd

arquivos = ['@mimimidiastweets.csv', '@nauticopetweets.csv', '@nautiluslinktweets.csv','@prefrecifetweets.csv', '@recifeordinariotweets.csv','@SantaCruzFCtweets.csv',
            '@sportrecifetweets.csv','Camaragibetweets.csv','CLTtweets.csv','Olindatweets.csv','Recifetweets.csv','UFPEtweets.csv','UFRPEtweets.csv']

dataframes = []
for arquivo in arquivos:
    df = pd.read_csv(arquivo)
    dataframes.append(df)
resultado = pd.concat(dataframes)

resultado.to_csv('arquivoResultado.csv', index=False)
resultado.to_csv('arquivoResultado2.csv', index=True)

print("terminado")