import os
import json
import requests
from flask import Flask, request, jsonify

# Carrega variáveis de ambiente do arquivo .env para desenvolvimento local
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Importações da LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase

# CHAIN RESPONSIBLE TO BUILD THE SQL QUERY BASED ON THE TABLE SCHEMA AND PROMPT
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# 1. SETUP INICIAL E VARIÁVEIS DE AMBIENTE
# ==============================================================================

# Carrega as chaves e senhas a partir das variáveis de ambiente
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
WHATSAPP_ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")

# Inicializa o modelo de linguagem
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, google_api_key=GOOGLE_API_KEY)

# Configura a conexão com o banco de dados de finanças
db_uri = "sqlite:///finance_bot.db"
db = SQLDatabase.from_uri(db_uri)


# ==============================================================================
# 2. FUNÇÕES AUXILIARES
# ==============================================================================

def get_schema(_):
    """Retorna o schema do banco de dados."""
    return db.get_table_info()

def clean_sql_query(query):
    """Limpa a query SQL removendo as marcações de markdown."""
    # Remove ```sql e ```
    cleaned_query = re.sub(r'^```sql\s*|\s*```$', '', query, flags=re.MULTILINE).strip()
    return cleaned_query

def run_query(query):
    """Executa uma query SQL no banco de dados após limpá-la."""
    cleaned_query = clean_sql_query(query)
    return db.run(cleaned_query)

def send_whatsapp_message(to_number, message_text):
    """Envia uma mensagem de texto para um número no WhatsApp."""
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "text": {"body": message_text},
    }
    url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status() # Lança um erro se a requisição falhar
    return response.json()


# ==============================================================================
# 3. LÓGICA DE CONSULTA (SELECT)
# ==============================================================================

# Prompt para gerar queries de consulta
query_template = """
Baseado no schema da tabela abaixo, escreva uma query SQL que responda à pergunta do usuário.
Responda apenas com a query SQL.

Schema: {schema}
Pergunta: {question}
SQL Query:
"""
query_prompt = ChatPromptTemplate.from_template(query_template)

# Prompt para gerar a resposta final em linguagem natural
response_template = """
Baseado no schema da tabela, na pergunta do usuário, na query SQL e na resposta do banco de dados, escreva uma resposta em linguagem natural e amigável.

Schema: {schema}
Pergunta: {question}
SQL Query: {query}
SQL Response: {response}
Resposta:
"""
response_prompt = ChatPromptTemplate.from_template(response_template)

# Chain para gerar o SQL de consulta
sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | query_prompt
    | llm
    | StrOutputParser()
)

# Chain completa para consultar e responder
query_chain = (
    RunnablePassthrough.assign(
        query=sql_chain
    ).assign(
        schema=get_schema,
        response=lambda variables: run_query(variables["query"])
    )
    | response_prompt
    | llm
    | StrOutputParser()
)


# ==============================================================================
# 4. LÓGICA DE INSERÇÃO (INSERT)
# ==============================================================================

# Prompt para gerar o SQL de inserção
insert_template = """
Baseado no schema da tabela a seguir, crie um comando SQL INSERT para adicionar o seguinte gasto: {question}.
Tente inferir a categoria do gasto (como 'Alimentação', 'Transporte', 'Lazer', etc.) a partir da descrição. Se a categoria não for clara, use 'Outros'.
Use a data e hora atuais para o campo data_hora (usando a função SQL NOW()) e defina o campo 'confirmado' como 0 (FALSE).

Schema: {schema}
SQL Query:
"""
insert_prompt = ChatPromptTemplate.from_template(insert_template)

# Prompt para a mensagem de confirmação
confirmation_template = """
Você é um assistente de finanças. O usuário pediu para adicionar um gasto, e o comando SQL foi executado com sucesso.
Escreva uma resposta curta e amigável confirmando que o gasto foi adicionado.

Pedido Original do Usuário: {question}
"""
confirmation_prompt = ChatPromptTemplate.from_template(confirmation_template)

# Chain para gerar o SQL de inserção
insert_sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | insert_prompt
    | llm
    | StrOutputParser()
)

# Chain completa para adicionar despesa e confirmar
add_expense_chain = (
    RunnablePassthrough.assign(
        query=insert_sql_chain
    ).assign(
        result=lambda variables: run_query(variables["query"])
    )
    | confirmation_prompt
    | llm
    | StrOutputParser()
)


# ==============================================================================
# 5. APLICAÇÃO FLASK (O SERVIDOR WEB)
# ==============================================================================

app = Flask(__name__)

@app.route('/webhook', methods=['GET'])
def verify_webhook():
    """Verifica o token do webhook para a Meta."""
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if request.args.get("hub.verify_token") == WHATSAPP_VERIFY_TOKEN:
            return request.args.get("hub.challenge"), 200
    return "Failed validation", 403

@app.route('/webhook', methods=['POST'])
def receive_message():
    """Recebe e processa as mensagens do WhatsApp."""
    data = request.get_json()
    
    try:
        # Extrai as informações importantes da mensagem
        message = data['entry'][0]['changes'][0]['value']['messages'][0]
        user_message = message['text']['body']
        user_phone_number = message['from']

        resposta_do_bot = ""
        
        # Lógica para decidir qual chain usar
        palavras_chave_insercao = ["adicione", "gastei", "registre", "paguei", "comprei"]
        if any(palavra in user_message.lower() for palavra in palavras_chave_insercao):
            resposta_do_bot = add_expense_chain.invoke({"question": user_message})
        else:
            resposta_do_bot = query_chain.invoke({"question": user_message})

        # Envia a resposta de volta para o usuário
        send_whatsapp_message(user_phone_number, resposta_do_bot)

    except (KeyError, IndexError, TypeError) as e:
        # Ignora eventos que não são mensagens do usuário (ex: status de entrega)
        print(f"Ignorando evento não-mensagem ou erro de parsing: {e}")
        pass

    return jsonify({"status": "ok"}), 200
