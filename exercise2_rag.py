from dotenv import load_dotenv
load_dotenv()

import os
import glob
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq


def load_text_files(folder: str) -> List[str]:
    paths = glob.glob(os.path.join(folder, "*.txt"))
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs


def chunk_text(text: str, chunk_words: int = 80, overlap_words: int = 20) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_words]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_words - overlap_words)
    return chunks


def build_index(chunks: List[str]) -> Tuple[TfidfVectorizer, any]:
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


def retrieve(query: str, chunks: List[str], vectorizer: TfidfVectorizer, matrix, k: int = 3) -> List[str]:
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idx = sims.argsort()[-k:][::-1]
    return [chunks[i] for i in top_idx]


def main():
    docs = load_text_files("docs")
    if not docs:
        print("No .txt files found in ./docs")
        return

    # Chunk all docs
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_text(d))

    vectorizer, matrix = build_index(all_chunks)

    question = input("Ask a question: ").strip()
    if not question:
        print("Error: question cannot be empty.")
        return

    top_chunks = retrieve(question, all_chunks, vectorizer, matrix, k=3)

    context = "\n\n---\n\n".join(top_chunks)

    prompt = f"""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say: "I don't know based on the provided context."

CONTEXT:
{context}

QUESTION:
{question}
"""

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    answer = llm.invoke(prompt).content

    print("\n=== Retrieved Context (Top 3) ===\n")
    for i, c in enumerate(top_chunks, 1):
        print(f"[Chunk {i}]\n{c}\n")

    print("\n=== Final Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
