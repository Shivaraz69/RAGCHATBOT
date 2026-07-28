import ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("🤖 Local RAG Chatbot (Ollama)")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    if not context.strip():
        print("No relevant information found.")
        continue

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.
If the answer is not present in the context, say:
"I couldn't find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        print("\nAnswer:\n")
        print(response["message"]["content"])

    except Exception as e:
        print("\nError:")
        print(e)