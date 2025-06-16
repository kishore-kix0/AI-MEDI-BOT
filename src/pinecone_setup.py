from pinecone_setup import index
from langchain_pinecone import PineconeVectorStore
from helper import download_huggingface_embeddings

embedding = download_huggingface_embeddings()

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding,
    text_key="text"
)

retriever = vectorstore.as_retriever()
