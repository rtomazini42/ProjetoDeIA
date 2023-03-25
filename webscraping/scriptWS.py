import requests
import os
import csv

# Definir as credenciais de acesso à API do Twitter
bearer_token = "" 
#se vc for minha colega Lidiane pode me pedir o bearer_token. Se você for outra pessoa por favor vá causar prejuízos pra Big Techs e não pra gente.




def busca(query, max_results=100):

		# Definir o endpoint da API do Twitter para busca de tweets
	search_url = "https://api.twitter.com/2/tweets/search/recent"

	# Configurar os cabeçalhos da solicitação
	headers = {
		"Authorization": f"Bearer {bearer_token}"
	}

	# Configurar os parâmetros da solicitação
	params = {
		"query": query,
		"max_results": max_results,
        "tweet.fields": "lang"
	}

	# Enviar a solicitação para a API do Twitter
	response = requests.get(search_url, headers=headers, params=params)
	print(response.json())
	# Verificar se a solicitação foi bem-sucedida
	if response.status_code == 200:
		# Extrair os dados dos tweets da resposta
		data = response.json().get("data")
		# Gravar tweets
		with open(query + "tweets.csv", "w", newline="", encoding="utf-8") as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(["id", "texto", "author_id", "sentimento"])
			for tweet in data:
				if tweet.get("lang") == "pt":
					print([tweet.get("id"), tweet.get("text"), tweet.get("author_id")])
					writer.writerow([tweet.get("id"), tweet.get("text"), tweet.get("author_id"),"vazio"])
	else:
		print("Erro ao buscar tweets:", response.status_code)

lista = ["Recife", "Pernambuco", "Camaragibe","Olinda", "UFRPE", "UFPE", "@prefrecife","APAC", "@SantaCruzFC", "@sportrecife", "@nauticope", "@recifeordinario","@nautiluslink","@mimimidias","CLT"]
for palavra in lista:
	busca(palavra)


