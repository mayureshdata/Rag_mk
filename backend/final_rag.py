import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone  # Pinecone integration from LangChain
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os


load_dotenv()  # Loads API keys from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
# pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")


# Load data
df = pd.read_csv("healthcare_data.csv")

# Optional: Clean and preprocess if there are NaN values or inconsistencies
df.fillna("", inplace=True)

# Concatenate relevant columns to create a context for each row
df['context'] = df.apply(lambda row: f"Name: {row['Name']}, Age: {row['Age']}, Gender: {row['Gender']}, "
                                     f"Blood Type: {row['Blood Type']}, Medical Condition: {row['Medical Condition']}, "
                                     f"Date of Admission: {row['Date of Admission']}, Doctor: {row['Doctor']}, "
                                     f"Hospital: {row['Hospital']}, Insurance Provider: {row['Insurance Provider']}, "
                                     f"Billing Amount: {row['Billing Amount']}, Room Number: {row['Room Number']}, "
                                     f"Admission Type: {row['Admission Type']}, Discharge Date: {row['Discharge Date']}, "
                                     f"Medication: {row['Medication']}, Test Results: {row['Test Results']}", axis=1)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Split and embed
documents = []
for context in df['context']:
    chunks = text_splitter.split_text(context)
    for chunk in chunks:
        documents.append({"text": chunk, "embedding": embeddings.embed_query(chunk)})


# Initialize Pinecone
pc = Pinecone(PINECONE_API_KEY)  # Pinecone instance
index = pc.Index(INDEX_NAME)  # Connect to Pinecone index
vector_store = PineconeVectorStore(index=index, embedding=embeddings)  # Vector store setup

# Define metadata schema for typical query use cases
# If the index doesn't exist, create it with the desired schema
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=len(documents[0]["embedding"]),
        metadata_config={
            "schema": {
                "age": "int",
                "gender": "str",
                "medical_condition": "str",
                "doctor": "str",
                "hospital": "str",
                "admission_date": "str",
                "discharge_date": "str"
            }
        }
    )

# Connect to the index
index = pinecone.Index(index_name)



# Prepare the data for upsert
upsert_data = []
for i, document in enumerate(documents):
    metadata = {
        "age": int(df.iloc[i]['Age']),
        "gender": df.iloc[i]['Gender'],
        "medical_condition": df.iloc[i]['Medical Condition'],
        "doctor": df.iloc[i]['Doctor'],
        "hospital": df.iloc[i]['Hospital'],
        "admission_date": df.iloc[i]['Date of Admission'],
        "discharge_date": df.iloc[i]['Discharge Date'],
    }
    upsert_data.append((str(i), document["embedding"], metadata))

# Upsert into Pinecone in batches (for efficiency)
batch_size = 100
for i in range(0, len(upsert_data), batch_size):
    index.upsert(upsert_data[i:i+batch_size])



# Prepare the data for upsert
upsert_data = []
for i, document in enumerate(documents):
    metadata = {
        "age": int(df.iloc[i]['Age']),
        "gender": df.iloc[i]['Gender'],
        "medical_condition": df.iloc[i]['Medical Condition'],
        "doctor": df.iloc[i]['Doctor'],
        "hospital": df.iloc[i]['Hospital'],
        "admission_date": df.iloc[i]['Date of Admission'],
        "discharge_date": df.iloc[i]['Discharge Date'],
    }
    upsert_data.append((str(i), document["embedding"], metadata))

# Upsert into Pinecone in batches (for efficiency)
batch_size = 100
for i in range(0, len(upsert_data), batch_size):
    index.upsert(upsert_data[i:i+batch_size])
