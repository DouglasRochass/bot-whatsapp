import os
import json
import re
import telebot # <-- ADICIONADO: Biblioteca para o bot do Telegram

# Carrega variáveis de ambiente do arquivo .env para desenvolvimento local
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Importações da LangChain (permanecem as mesmas)
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# 1. SETUP INICIAL E VARIÁVEIS DE AMBIENTE
# ==============================================================================

# Carrega as chaves e senhas a partir das variáveis de ambiente
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# REMOVIDO: Variáveis do WhatsApp não são mais necessárias
# ADICIONADO: Token para o bot do Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# Inicializa o modelo de linguagem
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=GOOGLE_API_KEY)

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
    cleaned_query = re.sub(r'^```sql\s*|\s*```$', '', query, flags=re.MULTILINE).strip()
    return cleaned_query

def run_query(query):
    """Executa uma query SQL no banco de dados após limpá-la."""
    cleaned_query = clean_sql_query(query)
    print(f"Executando Query: {cleaned_query}")
    return db.run(cleaned_query)



# ==============================================================================
# 3. LÓGICA DE CONSULTA (SELECT)
# ==============================================================================

query_template = """
Baseado no schema da tabela abaixo, escreva uma query SQL que responda à pergunta do usuário.
Responda apenas com a query SQL.
Schema: {schema}
Pergunta: {question}
SQL Query:
"""
query_prompt = ChatPromptTemplate.from_template(query_template)

response_template = """
Baseado no schema da tabela, na pergunta do usuário, na query SQL e na resposta do banco de dados, escreva uma resposta em linguagem natural e amigável. Escreva a data e o horário atuais no formato DD/MM/AAAA HH:MM:SS.
A cada descrição de gasto, separe por enter de uma compra para outra. utilize o formato "Descrição: [descrição], Valor: R$ [valor], Data e Hora: [data_hora] (no formato DD/MM/AAAA HH:MM:SS), Categoria: [categoria]".
Schema: {schema}
Pergunta: {question}
SQL Query: {query}
SQL Response: {response}
Resposta:
"""
response_prompt = ChatPromptTemplate.from_template(response_template)

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | query_prompt
    | llm
    | StrOutputParser()
)

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
# 4. LÓGICA DE INSERÇÃO
# ==============================================================================

insert_template = """
Baseado no schema da tabela a seguir, crie um comando SQL INSERT para adicionar o seguinte gasto: {question}.
Tente inferir a categoria do gasto (como 'Alimentação', 'Transporte', 'Lazer', etc.) a partir da descrição. Se a categoria não for clara, use 'Outros'.
Use a data e hora fornecida pela mensagem para o campo data_hora (caso ele não informar a data, utilize a função SQL datetime('now', 'localtime') para adicionar com o horário que a mensagem foi recebida) e defina o campo 'confirmado' como 0 (FALSE).
Schema: {schema}
SQL Query:
"""
insert_prompt = ChatPromptTemplate.from_template(insert_template)

confirmation_template = """
Você é um assistente de finanças. O usuário pediu para adicionar um gasto, e o comando SQL foi executado com sucesso.
Escreva uma resposta curta e amigável confirmando que o gasto foi adicionado. Caso possível, repita o valor e a descrição do gasto para confirmação.
Pedido Original do Usuário: {question}
"""
confirmation_prompt = ChatPromptTemplate.from_template(confirmation_template)

insert_sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | insert_prompt
    | llm
    | StrOutputParser()
)

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
# 5. LÓGICA DO BOT DO TELEGRAM
# ==============================================================================

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
print("Bot do Telegram conectado e aguardando mensagens...")

# Este 'handler' processa todas as mensagens de texto recebidas
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """Recebe e processa as mensagens do Telegram."""
    user_message = message.text
    print(f"Mensagem recebida de {message.from_user.first_name}: {user_message}")

    try:
        resposta_do_bot = ""
        
        # Lógica para decidir qual chain usar
        palavras_chave_insercao = ["adicione", "gastei", "registre", "paguei", "comprei"]
        if any(palavra in user_message.lower() for palavra in palavras_chave_insercao):
            resposta_do_bot = add_expense_chain.invoke({"question": user_message})
        else:
            resposta_do_bot = query_chain.invoke({"question": user_message})

        # Envia a resposta de volta para o usuário no Telegram
        bot.reply_to(message, resposta_do_bot)

    except Exception as e:
        print(f"Ocorreu um erro ao processar a mensagem: {e}")
        bot.reply_to(message, "Desculpe, ocorreu um erro. Não consegui processar sua solicitação.")


# ==============================================================================
# 6. INICIA O BOT
# ==============================================================================
if __name__ == '__main__':
    # Mantém o bot rodando e verificando por novas mensagens
    bot.polling(non_stop=True)
