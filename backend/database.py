import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from pathlib import Path
from langchain_community.document_loaders.csv_loader import CSVLoader

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "chatbot"

# print('\n',OPENAI_API_KEY)
# print('\n',PINECONE_API_KEY)
# print('\n',INDEX_NAME)

# Initialize OpenAI Embeddings and Pinecone
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
pc = Pinecone(PINECONE_API_KEY)

# Connect to the Pinecone index
index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
print(f'\nconnected to this index {INDEX_NAME} in pinecode database{vector_store}\n')


# Create a CSV loader
# def read_and_load_csv():
#     # Load the CSV file
#     loader = CSVLoader(file_path='healthcare_dataset.csv')
#     documents = loader.load()
#     documents
#     All_Entrys = [doc for doc in documents]


# # upserting the data in db in form of vectors
# def ingest_data_to_pinecone():
#     for doc in All_Entrys:
#         embedding = embeddings.embed_query(doc.page_content)
#         # Prepare data for upsert
#         upsert_data = {
#             "id": f"{doc.metadata['row']}",  # unique ID for each entry, can be row number or other unique identifier
#             "values": embedding,
#             "metadata": doc.metadata
#         }
#         # Upsert into Pinecone
#         index.upsert(vectors=[upsert_data])
#         print('\n\n data is upserted in to pinecode successfully')



'''
from langchain.embeddings import OpenAIEmbeddings

# upserting the data in db in form of vectors
for doc in age_87_documents:
    embedding = embeddings.embed_query(doc.page_content)
    # Prepare data for upsert
    upsert_data = {
        "id": f"{doc.metadata['row']}",  # unique ID for each entry, can be row number or other unique identifier
        "values": embedding,
        "metadata": doc.metadata
    }

    # Upsert into Pinecone
    index.upsert(vectors=[upsert_data])


## Querying
query = "Patient with Cancer admitted urgently"

# Generate embedding for the query
query_embedding = embeddings.embed_query(query)

# Perform similarity search
results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

# Extract matching documents
matched_documents = [result["metadata"] for result in results["matches"]]

## beutification    
messages = [
    {"role": "system", "content": "You are a medical assistant providing insights based on patient records."},
    {"role": "user", "content": f"The query is: '{query}'. Below are the details of the most relevant patient records:\n\n"}
]

for i, doc in enumerate(matched_documents, start=1):
    messages.append({"role": "user", "content": f"Record {i}:\n{doc.page_content}\n\n"})

messages.append({"role": "user", "content": "Please provide a summary and any notable insights based on the query and records."})

# Step 4: Generate Response with OpenAI's ChatCompletion
response =openai.chat.completions.create(
    model="gpt-3.5-turbo",  # or "gpt-4" if available
    messages=messages,
    max_tokens=1550,
    temperature=0.7
)

# Output the generated response
print(response.choices[0].message.content)
'''

def read_and_load_csv(file_path):
    """
    Reads a CSV file and returns the loaded documents.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        list: A list of documents loaded from the CSV file.
    """
    try:
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()
        # all_entries = [doc for doc in documents]
        all_entries = documents[:1000]
        print("CSV file loaded successfully.",len(all_entries))
        return all_entries
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {e}")
        return []

def ingest_data_to_pinecone(documents):
    """
    Embeds documents and upserts them into Pinecone.
    Args:
        documents (list): A list of documents to be embedded and upserted.
    """
    for doc in documents:
        try:
            # Generate embedding for the document's content
            embedding = embeddings.embed_query(doc.page_content)
            
            # Prepare data for upsert
            upsert_data = {
                "id": str(doc.metadata.get('row', '')),  # Ensure ID is a unique identifier
                "values": embedding,
                "metadata": doc.metadata
            }

            # Upsert data into Pinecone index
            index.upsert(vectors=[upsert_data])
            print(f"Document with ID {upsert_data['id']} upserted successfully.")
        
        except Exception as e:
            print(f"An error occurred while upserting document {doc.metadata.get('row', '')}: {e}")

# Main execution
if __name__ == "__main__":
    file_path = 'healthcare_dataset.csv'
    documents = read_and_load_csv(file_path)
    
    # if documents:
    #     ingest_data_to_pinecone(documents)
