# import chromadb
# chroma_client = chromadb.Client()

from langchain_chroma import Chroma #database local db
from langchain_openai import OpenAIEmbeddings # for convert data to vector
import os
from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Load the API key from the environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
# print(API_KEY)

file_path = "data/processed_data2.csv"

loader = CSVLoader(file_path=file_path)
data = loader.load()
# print(data[0])


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(data)

# print(len(splits))
# print(splits[1],'\n\n\n\n')
# print(splits[2])

## persist datat to db
vectordb = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# vectordb = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(),persist_directory='db')


### similarity search by text 
query = "which is expensive phone?"
docs = vectordb.similarity_search(query)
print('search by text ',docs[0].page_content)

### similarity search by vector
embedding_vector = OpenAIEmbeddings().embed_query(query)
docs = vectordb.similarity_search_by_vector(embedding_vector)
print('search by vector',docs[0].page_content)
# print(docs)


