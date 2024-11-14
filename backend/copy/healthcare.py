# from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
import openai
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import OpenAI
from pathlib import Path


### load the data 
# https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.csv_loader.CSVLoader.html#csvloader

loader = CSVLoader(file_path='healthcare_dataset.csv')
documents = loader.load()
documents
print(documents[0])

'''
List Comprehension Syntax: The general syntax of a list comprehension in Python is:
[expression for item in iterable if condition]
expression: This is what gets added to the new list. In your case, it is simply doc.
for item in iterable: This defines the loop, where item takes each value from the iterable (in this case, documents).
if condition: This is an optional filter that allows you to include only those items that satisfy the condition.
'''
All_Entrys = [doc for doc in documents]
print(All_Entrys)

# age_87_documents = [doc for doc in documents if 'Age: 87' in doc.page_content]
# age_above_40_documents=[doc for doc in documents if doc.page_content.split('Age: ')[1].split(',')[0] > '40']
# print(len(age_above_40_documents))
# Display the filtered documents
# for doc in age_87_documents:
#     print(doc,'\n')
#     print(doc.page_content)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("Pinecone_API_KEY")

# Initialize OpenAI Embeddings and Pinecone
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
pc = Pinecone(PINECONE_API_KEY)
index_name = "hospitaldemo"

# Connect to the Pinecone index
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# upserting the data in db in form of vectors
