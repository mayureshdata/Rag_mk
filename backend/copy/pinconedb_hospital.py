# import os
# from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langchain_community.llms import OpenAI
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from pathlib import Path
# import pandas as pd


# # from langchain.llms import OpenAI

# load_dotenv()
# directory_path = "hospital/"

# # Initialize OpenAI and Pinecone
# API_KEY = os.getenv("OPENAI_API_KEY")
# embeddings = OpenAIEmbeddings()
# Pinecone_API_KEY = os.getenv("Pinecone_API_KEY")
# print('Pinecone_API_KEY\n',Pinecone_API_KEY)
# pc = Pinecone(Pinecone_API_KEY)

# # print(pc)
# index_name = "hospitaldemo"
# # Check if the index exists, otherwise create it
# # if index_name not in pc.list_indexes():
# #     pc.create_index(index_name, dimension=1536, metric="cosine")

# # Connect to the Pinecone index
# index = pc.Index(index_name)
# print('index is -->>>\n',index)
# vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# # Step 1: Read all CSV files in the directory
# def read_all_csv_files(directory):
#     csv_files = []
#     for file in Path(directory).rglob("*.csv"):
#         df = pd.read_csv(file)
#         csv_files.append(df)
#     return csv_files

# # csv_dataframes = read_all_csv_files(directory_path)

# # vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())

# # Step 2: Process each row, create embeddings, and add them to Pinecone
# def upserting_data(csv_dataframes):
#     for df in csv_dataframes:
#         for _, row in df.iterrows():
#             # Convert row data to a single string (choose columns as needed)
#             row_text = " ".join(row.astype(str).tolist())
#             # Generate embeddings
#             embedding_vector = embeddings.embed_query(row_text)
#             # Generate a unique ID for each row (optional: you can use hash or UUID)
#             row_id = f"{hash(row_text)}"
#             # Upsert the embedding to Pinecone
#             index.upsert([(row_id, embedding_vector, {"text": row_text})])
# # upserting_data(csv_dataframes)
# print(f"Data from all CSV files has been added to the Pinecone index '{index_name}'.")

# ### Optional: Define a retrieval function to query Pinecone
# # def retrieve_similar_text(query, k=4):
# #     query_embedding = embeddings.embed_query(query)
# #     # print(query_embedding)
# #     results = vector_store.similarity_search(query=query, k=k)
# #     # print(results)
# #     return results


# # Optional: Define a retrieval function to query Pinecone
# # def retrieve_similar_text(query, k=3):
# #     query_embedding = embeddings.embed_query(query)
# #     results = index.query(vector=query_embedding, top_k=k, include_metadata=True)
# #     return results["matches"]

# ### Test a query
# ##query = "how many beds maintained by Southern Railway?"
# ##query = "how many beds maintained by North Western Railway?"
# ##query = "how many beds and Hospitals in All India?"
# # matches = retrieve_similar_text(query)
# # # for doc in matches:
# #     # print(f"* {doc.page_content} [{doc.metadata}]")
# #     # print('matches\n\n',matches[0].page_content)
# # for match in matches:
# #     print(f"Match score: {match['score']}, Text: {match['metadata']['text']}")


# from langchain_openai import ChatOpenAI
# from langchain_experimental.agents import create_csv_agent

# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# agent_executor = create_csv_agent(
#     llm,
#     "AYUSHHospitals.csv",
#     agent_type="openai-tools",
#     verbose=True
# )
# output = agent_executor.invoke("How many total number of Ministry of Defence?")
# print('output--->>>>', output)

# # Initialize OpenAI with a specific model (e.g., GPT-4)
# # llm = OpenAI(model="gpt-4", api_key=API_KEY)

# # def generate_coherent_response(query, matches):
# #     # Construct context from the retrieved metadata
# #     context = "\n".join([match["metadata"]["text"] for match in matches])

# #     # Construct the prompt for the LLM
# #     prompt = f"Based on the following context, answer the question: '{query}'\n\nContext:\n{context}"

# #     # Generate the response using the LLM
# #     response = llm.generate(prompt)
# #     return response

# # # Example use
# # query = "how many beds and Hospitals in All India?"
# # matches = retrieve_similar_text(query)
# # response = generate_coherent_response(query, matches)
# # print(response)



# # from langchain_openai import ChatOpenAI
# # from langchain.chains import create_retrieval_chain
# # from langchain.chains.combine_documents import create_stuff_documents_chain
# # from langchain import hub
# # # from langchain.chains import RetrievalQA


# # # https://python.langchain.com/api_reference/langchain/hub.html#module-langchain.hub
# # # https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval.create_retrieval_chain.html#create-retrieval-chain
# # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
# # retriever = vector_store.as_retriever()

# # # print("retriever information",retriever)

# # llm = ChatOpenAI(
# #     openai_api_key=API_KEY,
# #     # model_name='gpt-4o-mini',
# #     model_name='gpt-3.5-turbo',
# #     # model_name='gpt-4',
# #     temperature=0.2,
# # )

# # # https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval.create_retrieval_chain.html#create-retrieval-chain
# # combine_docs_chain = create_stuff_documents_chain(
# #     llm, retrieval_qa_chat_prompt
# # )
# # retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
# # # print('information about retrieval_chain ',retrieval_chain)

# # query_by_user = "how many beds maintained by Southern Railway?"

# # # https://python.langchain.com/docs/versions/migrating_chains/retrieval_qa/
# # answer_with_knowledge = retrieval_chain.invoke({"input": query_by_user})

# # print("\nAnswer with knowledge:\n\n", answer_with_knowledge['answer'])
# # print("\nContext used:\n", answer_with_knowledge['context'])
# # print("\n")
# # time.sleep(2)



# from langchain_community.llms import OpenAI
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from pathlib import Path
# import pandas as pd
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import OpenAI
from pathlib import Path

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("Pinecone_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENV")  # For example, 'us-east1-gcp'

# Initialize OpenAI Embeddings and Pinecone
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
pc = Pinecone(PINECONE_API_KEY)
index_name = "hospitaldemo"


# Connect to the Pinecone index
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Function to read all CSV files in a directory and return DataFrames
def read_all_csv_files(directory):
    csv_files = []
    for file in Path(directory).rglob("*.csv"):
        df = pd.read_csv(file)
        csv_files.append(df)
    return csv_files

# Ingest CSV data into Pinecone
def ingest_data_to_pinecone(directory_path):
    csv_dataframes = read_all_csv_files(directory_path)
    for df in csv_dataframes:
        for _, row in df.iterrows():
            # Convert each row to a single text string
            row_text = " ".join(row.astype(str).tolist())
            # Generate embedding for the row text
            embedding_vector = embeddings.embed_query(row_text)
            # Create a unique ID for the row
            row_id = f"{hash(row_text)}"
            # Upsert the embedding with metadata
            index.upsert([(row_id, embedding_vector, {"text": row_text})])

# Query Pinecone and generate a response using an LLM
def query_pinecone(query, k=3):
    # Convert query to embedding
    query_embedding = embeddings.embed_query(query)
    # Perform similarity search in Pinecone
    results = index.query(vector=query_embedding, top_k=k, namespace="hospitaldemo",include_metadata=True)
    
    # Initialize LLM for response generation
    llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
    
    # Prepare context from retrieved results
    context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
    
    # Construct the prompt for the LLM
    prompt = f"Based on the following context, answer the question: '{query}'\n\nContext:\n{context}"
    
    # Generate the response
    response = llm.generate(prompt)
    return response

# Main execution
if __name__ == "__main__":
    # Directory containing CSV files
    directory_path = "hospital/"
    
    # Ingest CSV data into Pinecone
    ingest_data_to_pinecone(directory_path)
    print("Data ingestion complete. All CSV data has been indexed in Pinecone.")
    
    # Example query
    user_query = "how many total hospitals in india?"
    answer = query_pinecone(user_query)
    print("Answer:", answer)
