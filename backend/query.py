from langchain_community.llms import OpenAI
import os
from dotenv import load_dotenv
from database import vector_store,index,INDEX_NAME
from langchain_openai import OpenAIEmbeddings ,OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)

# =========================================================

# def query_pinecone(query, vector_store=vector_store, k=3):
#     query_vector = vector_store.embeddings.embed_query(query)
#     # results = vector_store.similarity_search(query_vector, top_k=k, include_metadata=True)
#     results = index.query(vector=query_vector, top_k=k, namespace=INDEX_NAME,include_metadata=True)
#     context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
#     prompt = f"Based on the following context, answer the question: '{query}'\n\nContext:\n{context}"
#     response = llm.generate(prompt)
#     return response

# query_pinecone("how many beds maintained by Southern Railway?")

# =========================================================

## Querying
query = "Patient with Cancer admitted urgently"
def chat_with_user(query):
    # Generate embedding for the query
    query_embedding = embeddings.embed_query(query)
    # print(query_embedding)
    # Search for similar vectors in the vector store
    results = vector_store.similarity_search(query=query_embedding)
    print('result by similarity search by vector',results)

output = chat_with_user("Patient with Cancer admitted urgently")
print(output)
# =========================================================

# =========================================================

## Querying
# query = "Patient with Cancer admitted urgently"
# def chat_with_user(query):
#     # Generate embedding for the query
#     query_embedding = embeddings.embed_query(query)
#     print(query_embedding)
    # Perform similarity search
    # results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    # print(results)

    # matching_results = vector_store.similarity_search(query=query,k=2)
    # Extract the row numbers of the matched documents      
    # matching_rows = [int(result["metadata"]["row"]) for result in results["matches"]]
    # print(matching_rows)
    # # Retrieve documents by matching rows in loaded csv but we want to retrieve from pinecone
    # matched_documents = [doc for doc in results if int(doc.metadata["row"]) in matching_rows]
    # # matched_documents = [doc for doc in documents if int(doc.metadata["row"]) in matching_rows]
    # matched_documents

    # # ## beutification    
    # messages = [
    #     {"role": "system", "content": "You are a medical assistant providing insights based on patient records."},
    #     {"role": "user", "content": f"The query is: '{query}'. Below are the details of the most relevant patient records:\n\n"}
    # ]

    # for i, doc in enumerate(matched_documents, start=1):
    #     messages.append({"role": "user", "content": f"Record {i}:\n{doc.page_content}\n\n"})

    # messages.append({"role": "user", "content": "Please provide a summary and any notable insights based on the query and records."})

    # ###Step 4: Generate Response with OpenAI's ChatCompletion
    # response =OpenAI.chat.completions.create(
    #     # model="gpt-3.5-turbo",  # or "gpt-4" if available
    #     model="gpt-4",  # or "gpt-4" if available
    #     messages=messages,
    #     max_tokens=1550,
    #     temperature=0.5
    # )
    # # Output the generated response
    # print(response.choices[0].message.content)

# output = chat_with_user("Patient with Cancer admitted urgently")
# print(output)
# =========================================================

# query = "Patient with Cancer admitted urgently"
# def chat_with_user(query):
#     try:
#         # Generate embedding for the query
#         query_embedding = embeddings.embed_query(query)

#         # Perform similarity search
#         results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

#         # Extract matching documents
#         matched_documents = [result["metadata"] for result in results["matches"]]
#         print('Matched Documents:\n', matched_documents)

#         # Prepare messages for the AI model
#         messages = create_message_payload(query, matched_documents)

#         # Generate response using OpenAI's ChatCompletion
#         response = generate_response(messages)

#         # Output the generated response
#         print_response(response)

#     except Exception as e:
#         print(f"An error occurred: {e}")

# def create_message_payload(query, matched_documents):
#     """Creates the message payload for the AI model."""
#     messages = [
#         {"role": "system", "content": "You are a medical assistant providing insights based on patient records."},
#         {"role": "user", "content": f"The query is: '{query}'. Below are the details of the most relevant patient records:\n\n"}
#     ]

#     for i, doc in enumerate(matched_documents, start=1):
#         messages.append({"role": "user", "content": f"Record {i}:\n{doc.page_content}\n\n"})

#     messages.append({"role": "user", "content": "Please provide a summary and any notable insights based on the query and records."})
    
#     return messages

# def generate_response(messages):
#     """Generates a response from the OpenAI model."""
#     return OpenAI.chat.completions.create(
#         model="gpt-4",  # or "gpt-4" if available
#         # model="gpt-3.5-turbo",  # or "gpt-4" if available
#         messages=messages,
#         max_tokens=1550,
#         temperature=0.5
#     )

# def print_response(response):
#     """Prints the response from the AI model."""
#     print(response.choices[0].message.content)

# # Execute the chat function with the specified query
# output = chat_with_user(query)

# =========================================================


