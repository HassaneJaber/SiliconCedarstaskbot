from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise and practical."),
    ("human", "Create a {minutes}-minute study plan for {topic} in {n} bullet points.")
])

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"minutes": 30, "topic": "LangChain basics", "n": 6})
print(result)
