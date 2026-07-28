from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Read PDF
pdf = PdfReader("basic.pdf")

text = ""
for page in pdf.pages:
    if page.extract_text():
        text += page.extract_text()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Total Chunks:", len(chunks))

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS database
db = FAISS.from_texts(chunks, embedding_model)

# Save database
db.save_local("faiss_index")

print("FAISS index created successfully!")