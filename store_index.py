from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone (NEW SDK)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index name
index_name = "medchat"

# Create index if it does NOT exist
existing_indexes = [i["name"] for i in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # HuggingFace embeddings dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to index
index = pc.Index(index_name)

# Load & process documents
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Store embeddings in Pinecone
LangchainPinecone.from_texts(
    texts=[t.page_content for t in text_chunks],
    embedding=embeddings,
    index=index,
    text_key="text"
)

print("✅ Documents successfully indexed in Pinecone")
