import os
import pandas as pd
from dotenv import load_dotenv
import pinecone
from langchain_openai import OpenAIEmbeddings,OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Initialize Pinecone
file_path = 'healthcare_dataset.csv'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENV")
index_name = "chatbot"
# Create or connect to an index
pc = Pinecone(PINECONE_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
# Initialize OpenAI LLM
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

# Connect to the Pinecone index
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize the ChatOpenAI model
llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")

# def load_csv(file_path):
#     return pd.read_csv(file_path)

# def load_pdf(file_path):
#     import pdfplumber
#     with pdfplumber.open(file_path) as pdf:
#         return "\n".join(page.extract_text() for page in pdf.pages)

# def load_txt(file_path):
#     with open(file_path, 'r') as file:
#         return file.read()

def create_embeddings(text):
    from langchain_openai import OpenAIEmbeddings
    embedding_model = OpenAIEmbeddings()
    return embedding_model.embed_query(text)

# def upsert_data(embeddings, metadata):
#     index.upsert(vectors=[(str(i), embeddings[i], metadata[i]) for i in range(len(embeddings))])

def retrieve(query):
    embedding = create_embeddings(query)
    results = index.query(vector=embedding, top_k=5, include_metadata=True)
    return results

def generate_response(matches, question):
    # Initialize an empty list to hold context strings
    contexts = []
    
    # Extract relevant information from matches
    for match in matches['matches']:
        metadata = match['metadata']
        # Construct a context string for each match
        context = (
            f"Name: {metadata['name']}, "
            f"Age: {metadata['age']}, "
            f"Admission Date: {metadata['admission_date']}, "
            f"Medical Condition: {metadata['medical_condition']}, "
            f"Billing Amount: ${metadata['billing_amount']:.2f}"
        )
        contexts.append(context)

    # Join all context strings into one single string
    context_string = "\n".join(contexts)

    # Create messages for the chat model
    messages = [
        {"role": "system", "content": "You are an assistant that provides medical information."},
        {"role": "user", "content": context_string},
        {"role": "user", "content": question}
    ]

    # Invoke the LLM with the formatted messages
    response = llm(messages)
    
    # Accessing the content of the response correctly
    return response.content if hasattr(response, 'content') else response

# Example usage to load data and generate responses
if __name__ == "__main__":
    # Load your data (CSV, PDF, TXT)
    # For example:
    # data_text = load_csv("data.csv")  # Load your CSV data
    
    # Assuming you have already loaded and processed your data into text format:
    
    # Create embeddings and upsert them into Pinecone (this should be done once)
    # embeddings = create_embeddings(data_text)
    # upsert_data(embeddings, metadata_list)  # metadata_list should be a list of metadata dictionaries
    
    # Example query to retrieve matches from Pinecone
    user_query = "What can you tell me about this patient?"
    
    matches = retrieve(user_query)  # Retrieve matches based on user query
    
    if matches['matches']:
        answer = generate_response(matches, user_query)
        print("Bot Response:", answer)
    else:
        print("No relevant information found.")