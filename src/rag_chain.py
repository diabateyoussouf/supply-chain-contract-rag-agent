# Step 1: Import necessary libraries
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,TextLoader,PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()


loaders = PyPDFDirectoryLoader("/home/diabate/Bureau/supply-chain-contract-rag-agent/data/pdf/full_contract_pdf/full_contract_pdf/Part_I/Affiliate_Agreements/", glob="*.pdf")   


# Step 2: Load and split documents

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=200)

chunks = []
for loader in loaders.load():
    chunks.append(loader.load_and_split(text_splitter))

# Step 3: Create embeddings and store in Chroma vector store
embedding_model  = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = Chroma.from_documents(
    documents=chunks, embedding=OpenAIEmbeddings(
        model="text-embedding-3-small", 
        embedding_model=embedding_model,
        collection_name="contracts"))

# Step 4: Save the vector store to disk

retriever=vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5}
)

retriever_chunks = retriever.invoke("What are the key terms of the affiliate agreements?")

print(retriever_chunks)

