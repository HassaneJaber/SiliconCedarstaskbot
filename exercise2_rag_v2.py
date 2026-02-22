from dotenv import load_dotenv
load_dotenv()

import os
import glob
from datetime import datetime
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq


def load_text_files(folder: str) -> List[Tuple[str, str]]:
    """Return list of (filename, content)."""
    paths = glob.glob(os.path.join(folder, "*.txt"))
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(p), f.read()))
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


def retrieve_with_scores(query: str, chunks: List[str], vectorizer: TfidfVectorizer, matrix, k: int = 3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idx = sims.argsort()[-k:][::-1]
    return [(chunks[i], float(sims[i])) for i in top_idx]


def main():
    docs = load_text_files("docs")
    if not docs:
        print("No .txt files found in ./docs")
        return

    # Chunk all docs (store mapping for "sources")
    chunk_bank = []
    for fname, content in docs:
        for c in chunk_text(content):
            chunk_bank.append((fname, c))

    chunks_only = [c for _, c in chunk_bank]
    vectorizer, matrix = build_index(chunks_only)

    question = input("Ask a question: ").strip()
    if not question:
        print("Error: question cannot be empty.")
        return

    top = retrieve_with_scores(question, chunks_only, vectorizer, matrix, k=3)

    # Build context with separators
    context = "\n\n---\n\n".join([t[0] for t in top])

    prompt = f"""You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say: "I don't know based on the provided context."
At the end, add a short line: Sources: [kb.txt] (or the filenames you used).

CONTEXT:
{context}

QUESTION:
{question}
"""

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    answer = llm.invoke(prompt).content

    print("\n=== Retrieved Context (Top 3) ===\n")
    for i, (chunk, score) in enumerate(top, 1):
        print(f"[Chunk {i}] score={score:.3f}\n{chunk}\n")

    print("\n=== Final Answer ===\n")
    print(answer)

    # Save to logs
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join("logs", f"rag_run_{ts}.txt")

    # Find source filenames for the retrieved chunks
    used_files = set()
    for chunk, _ in top:
        for fname, c in chunk_bank:
            if c == chunk:
                used_files.add(fname)
                break

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Question: {question}\n\n")
        f.write("Top Chunks:\n")
        for i, (chunk, score) in enumerate(top, 1):
            f.write(f"\n[{i}] score={score:.3f}\n{chunk}\n")
        f.write("\nAnswer:\n")
        f.write(answer + "\n")
        f.write("\nSources:\n")
        f.write(", ".join(sorted(used_files)) + "\n")

    print(f"\nSaved log to: {out_file}")


if __name__ == "__main__":
    main()
