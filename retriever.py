from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone()
index_name = "medical-assistant"

def create_index():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


def load_pdf():
    loader = PyPDFLoader('medical_book.pdf')
    documents = loader.load()
    print(documents)
    return documents

def get_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
    vector_store = PineconeVectorStore.from_existing_index(index_name, embeddings)
    retriever = vector_store.as_retriever()
    return retriever

def split_docs(documents: List[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(documents)
    return splits


def embed_docs():
    print("Embedding in-progress ..")
    documents = load_pdf()
    splits = split_docs(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
    vector_store = PineconeVectorStore.from_documents(splits, embeddings, index_name=index_name)
    print("Embedding Done")
    return vector_store


if __name__ == '__main__':
    print(get_retriever().invoke('acne'))
