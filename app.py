from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from pinecone import Pinecone
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

# Load Pinecone API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone (NEW SDK)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index name (must already exist in Pinecone dashboard)
index_name = "medchat"
index = pc.Index(index_name)

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Load existing Pinecone index into LangChain
docsearch = LangchainPinecone(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Prompt setup
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

# Load LLaMA-2 model
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        "max_new_tokens": 512,
        "temperature": 0.8
    }
)

# Retrieval QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    result = qa({"query": msg})
    return result["result"]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
