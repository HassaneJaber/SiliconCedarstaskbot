from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")
resp = llm.invoke("Write a one-line fun fact about Beirut.")
print(resp.content)
