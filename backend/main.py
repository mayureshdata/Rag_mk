# from fastapi import FastAPI, UploadFile, File
from backend.database import ingest_data_from_csv
from query import query_pinecone
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from backend.database import embeddings,vector_store
from backend.query import llm

load_dotenv()

# app = FastAPI()

# @app.post("/upload-csv/")
# async def upload_csv(file: UploadFile = File(...)):
#     file_location = f"temp/{file.filename}"
#     with open(file_location, "wb+") as f:
#         f.write(file.file.read())
#     ingest_data_from_csv("temp")
#     return {"message": "CSV file ingested successfully"}

# @app.get("/query/")
# async def query_api(query: str):
#     response = query_pinecone(query, vector_store)
#     return {"response": response}


# Initialize Flask app
app = Flask(__name__)


# Endpoint for querying the RAG-based chatbot
@app.route("/query", methods=["POST"])
def query():
    # Parse the user query from request data
    user_data = request.get_json()
    user_query = user_data.get("query")
    
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    # Step 1: Embed the query and search similar documents in Pinecone
    query_vector = embeddings.embed_query(user_query)
    results = vector_store.index.query(query_vector, top_k=3, include_metadata=True)
    
    # Step 2: Aggregate context from retrieved documents
    context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
    
    # Step 3: Generate response using the language model
    prompt = f"Based on the following context, answer the question: '{user_query}'\n\nContext:\n{context}"
    generated_response = llm.generate(prompt)
    
    return jsonify({"response": generated_response})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True,port=5000)




    # =============================================

    '''
    from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

# API Keys and Configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")

# Initialize Flask app
app = Flask(__name__)

# Initialize embeddings and Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Check if the Pinecone index exists; create if not
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(name=INDEX_NAME, dimension=1536, metric="cosine")

index = pinecone.Index(INDEX_NAME)
vector_store = Pinecone(index=index, embedding=embeddings)

# Initialize OpenAI LLM for generating responses
llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)

# Endpoint for querying the RAG-based chatbot
@app.route("/query", methods=["POST"])
def query():
    # Parse the user query from request data
    user_data = request.get_json()
    user_query = user_data.get("query")
    
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    # Step 1: Embed the query and search similar documents in Pinecone
    query_vector = embeddings.embed_query(user_query)
    results = vector_store.index.query(query_vector, top_k=3, include_metadata=True)
    
    # Step 2: Aggregate context from retrieved documents
    context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
    
    # Step 3: Generate response using the language model
    prompt = f"Based on the following context, answer the question: '{user_query}'\n\nContext:\n{context}"
    generated_response = llm.generate(prompt)
    
    return jsonify({"response": generated_response})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)

    '''