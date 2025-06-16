from src.helper import download_huggingface_embeddings, load_pdf_file, text_split
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


load_dotenv()


PINECONE_API_KEY=os.environ.get('pcsk_4dRr3v_PyJzmcQsgBWkSb9AL9c3rMb1QDVTBy128Mbo1uQqezdCXKER2WtwsGy3iSHNNni')
os.environ["PINECONE_API_KEY"] = "pcsk_4dRr3v_PyJzmcQsgBWkSb9AL9c3rMb1QDVTBy128Mbo1uQqezdCXKER2WtwsGy3iSHNNni"

#load the files
extracted_data = load_pdf_file(data='C:\medical\Data')
text_chunks = text_split(extracted_data)
embeddings = download_huggingface_embeddings()

pc = Pinecone(api_key="pcsk_4dRr3v_PyJzmcQsgBWkSb9AL9c3rMb1QDVTBy128Mbo1uQqezdCXKER2WtwsGy3iSHNNni")
# Define index name and configuration
index_name = "medibot-v2"

# Create the index
pc.create_index(
    name=index_name,
    dimension=384,  # must match your embedding size
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Embed each chunk and upsert the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)
