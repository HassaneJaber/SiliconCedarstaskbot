from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def main():
    topic = input("Enter a topic to study: ").strip()
    if not topic:
        print("Error: topic cannot be empty.")
        return

    minutes_str = input("How many minutes do you have? (e.g., 30): ").strip()
    if not minutes_str.isdigit():
        print("Error: minutes must be a number like 30.")
        return
    minutes = int(minutes_str)

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise and practical."),
        ("human", "Create a {minutes}-minute study plan for {topic} in {n} bullet points.")
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"minutes": minutes, "topic": topic, "n": 8})

    print("\n--- Study Plan ---\n")
    print(result)

    # Save output to a file (for proof + portfolio)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"study_plan_{ts}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Topic: {topic}\nMinutes: {minutes}\n\n")
        f.write(result)

    print(f"\nSaved to: {filename}")

if __name__ == "__main__":
    main()
