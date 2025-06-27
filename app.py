from flask import Flask, render_template, request
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Together
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
from src.helper import download_huggingface_embeddings
from src.helper import load_pdf_file, text_split
from langchain_together import Together 
import os

# Load environment variables

load_dotenv()

# Get keys from .env or manually assign
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') or "your_pinecone_key"
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY') or "f99e4e137a324ca884a38408a1991c00c7ec67fbe53b5fce65ddb317e71c5a8b"

# Set environment for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

app = Flask(__name__)

# === Load and process PDF ===
extracted_data = load_pdf_file(data='C:/medical/Data')
text_chunks = text_split(extracted_data)

# === Embeddings and Vector Store ===
embeddings = download_huggingface_embeddings()
index_name = "medibot-v2"

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# === LLM (Together) Setup ===
llm = Together(
    temperature=0.4,
    max_tokens=300,
    model="mistralai/Mistral-7B-Instruct-v0.2",  # ✅ Serverless & Free model
    api_key=TOGETHER_API_KEY
)

# === Prompt + RAG Chain ===
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# === Flask Routes ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)  # DEBUG

    response = rag_chain.invoke({"input": msg})
    print("AI Response:", response["answer"])  # DEBUG

    return str(response["answer"])


if __name__ == "__main__":
    app.run("0.0.0.0", port=8080, debug=True)
