from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")

resp = llm.invoke("Say 'connected' if you can read this.")
print(resp.content)
