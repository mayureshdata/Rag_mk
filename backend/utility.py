import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings,OpenAI
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from pathlib import Path
from langchain_community.document_loaders.csv_loader import CSVLoader


# Load environment variables
load_dotenv()


file_path = 'healthcare_dataset.csv'
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

# Initialize OpenAI LLM
# llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

# Connect to the Pinecone index
index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
print(f'\nconnected to this index {INDEX_NAME} in pinecode database{vector_store}\n')
# loader = CSVLoader(file_path=file_path)
# documents = loader.load()
# # Take only the first 1000 records
# all_entries = documents[:10]
# ## Take all records in csv
# # all_entries = [doc for doc in documents]
# print("CSV file loaded successfully.",len(all_entries))
# print("CSV file loaded successfully.",all_entries[Document])

# from some_embeddings_module import Embeddings  # Replace with correct module


# Initialize necessary objects
# embeddings = Embeddings()  # Make sure Embeddings is properly configured
# index = PineconeIndex()  # Ensure PineconeIndex is properly connected to Pinecone

def read_and_load_csv(file_path, limit=5):
    """
    Reads a CSV file and returns the loaded documents, limited to a specific number of entries.

    Args:
        file_path (str): The path to the CSV file.
        limit (int): The maximum number of records to return.

    Returns:
        list: A list of documents loaded from the CSV file, limited by the `limit` parameter.
    """
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    return documents[:limit]

def sort_documents(documents, sort_key="Billing Amount"):
    """
    Sorts documents by a specified field within the page content.

    Args:
        documents (list): The list of documents to sort.
        sort_key (str): The key by which to sort the documents.

    Returns:
        list: Sorted documents based on the specified field.
    """
    def get_sort_value(doc):
        lines = doc.page_content.splitlines()
        for line in lines:
            if sort_key in line:
                return float(line.split(": ")[1])  # Adjust based on the expected value type
        return 0

    return sorted(documents, key=get_sort_value, reverse=True)

def ingest_data_to_pinecone(documents):
    """
    Embeds and upserts sorted documents into Pinecone.

    Args:
        documents (list): A list of documents to embed and upsert.
    """
    for doc in documents:
        try:
            # Generate embedding for the document's content
            embedding = embeddings.embed_query(doc.page_content)
            print(doc)
            # Prepare structured metadata from page content
            # metadata = {
            #     "name": doc.page_content.split("\n")[0].split(": ")[1],  # Extract Name
            #     "age": int(doc.page_content.split("\n")[1].split(": ")[1]),  # Extract Age
            #     "medical_condition": doc.page_content.split("Medical Condition: ")[1].split("\n")[0],
            #     "admission_date": doc.page_content.split("Date of Admission: ")[1].split("\n")[0],
            #     "billing_amount": float(doc.page_content.split("Billing Amount: ")[1].split("\n")[0]),
            #     "row": doc.metadata["row"]
            # }


            # Safely parse the page content
            content_lines = doc.page_content.split("\n")
            metadata = {
            "text": {
                "name": content_lines[0].split(": ")[1].strip(),
                "age": int(content_lines[1].split(": ")[1].strip()),
                "gender": content_lines[2].split(": ")[1].strip(),
                "blood_type": content_lines[3].split(": ")[1].strip(),
                "medical_condition": content_lines[4].split(": ")[1].strip(),
                "admission_date": content_lines[5].split(": ")[1].strip(),
                "doctor": content_lines[6].split(": ")[1].strip(),
                "hospital": content_lines[7].split(": ")[1].strip(),
                "insurance_provider": content_lines[8].split(": ")[1].strip(),
                "billing_amount": float(content_lines[9].split(": ")[1].strip()),
                "room_number": content_lines[10].split(": ")[1].strip(),
                "admission_type": content_lines[11].split(": ")[1].strip(),
                "discharge_date": content_lines[12].split(": ")[1].strip(),
                "medication": content_lines[13].split(": ")[1].strip(),
                "test_results": content_lines[14].split(": ")[1].strip(),
                "row": doc.metadata.get("row"),
                "source": doc.metadata.get("source")
                }
            }

            # metadata = {
            # "name": content_lines[0].split(": ")[1].strip(),
            # "age": int(content_lines[1].split(": ")[1].strip()),
            # "gender": content_lines[2].split(": ")[1].strip(),
            # "blood_type": content_lines[3].split(": ")[1].strip(),
            # "medical_condition": content_lines[4].split(": ")[1].strip(),
            # "admission_date": content_lines[5].split(": ")[1].strip(),
            # "doctor": content_lines[6].split(": ")[1].strip(),
            # "hospital": content_lines[7].split(": ")[1].strip(),
            # "insurance_provider": content_lines[8].split(": ")[1].strip(),
            # "billing_amount": float(content_lines[9].split(": ")[1].strip()),
            # "room_number": content_lines[10].split(": ")[1].strip(),
            # "admission_type": content_lines[11].split(": ")[1].strip(),
            # "discharge_date": content_lines[12].split(": ")[1].strip(),
            # "medication": content_lines[13].split(": ")[1].strip(),
            # "test_results": content_lines[14].split(": ")[1].strip(),
            # "row": doc.metadata.get("row"),
            # "source": doc.metadata.get("source")
            # }
            # Prepare data for upsert
            upsert_data = {
                "id": str(doc.metadata.get("row", "")),  # Unique identifier
                "values": embedding,
                "metadata": metadata
            }

            # Upsert data into Pinecone index
            index.upsert(vectors=[upsert_data])
            print(f"Document with ID {upsert_data['id']} upserted successfully.")

        except Exception as e:
            print(f"Error upserting document {doc.metadata.get('row', '')}: {e}")

def retrieve_data_from_pinecone(user_query, top_k=5, metadata_filter=None):
    """
    Retrieves the most similar documents from Pinecone based on the user query.

    Args:
        user_query (str): The textual query input by the user.
        top_k (int): The number of top results to retrieve.
        metadata_filter (dict): Optional metadata filters to refine search.

    Returns:
        list: Retrieved documents with metadata.
    """
    try:
        # Convert the user query into an embedding
        query_embedding = embeddings.embed_query(user_query)

        # Query Pinecone for similar documents
        query_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=metadata_filter,  # Use metadata_filter to narrow down results (optional)
            include_metadata=True  # Ensures metadata is returned with the results
        )
        # print('query_response',query_response)
        return [match['metadata'] for match in query_response['matches']]
        # return query_response
        # results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        # print('\n\nresults',results)
        # return [match['metadata'] for match in results['matches']]

        # context = "\n".join([match["metadata"] for match in query_response["matches"]])
        # print('\ncontext query_response',context)
        # return context
    
        # Process and return the query response
        # retrieved_documents = []
        # for result in query_response["matches"]:
        #     retrieved_document = {
        #         "id": result["id"],
        #         "score": result["score"],
        #         "metadata": result["metadata"]
        #     }
        #     retrieved_documents.append(retrieved_document)

        # return retrieved_documents
        

    except Exception as e:
        print(f"An error occurred during retrieval: {e}")
        return []

# ==================================
from langchain_core.prompts import ChatPromptTemplate
## just search about hub is for ready made template

# prompt_template = ChatPromptTemplate.from_template("""You are an AI healthcare assistant. Use the information provided in the context to answer the user’s question thoroughly and accurately. Approach the answer step-by-step to ensure completeness and clarity.
                                                   
# Context Information:
# {context}

# User Question:
# {input}

# Please provide a detailed and insightful response:""")
# print('\nprompt created by ChatPromptTemplate',prompt_template)
# prompt_template = ChatPromptTemplate.from_template("""
# Answer the following question based only on the provided context. mainly focued on the data set is for healthcare,
# Think step by step before providing a detailed answer.
# {context}
# Question:{input}
# """)
# print('\nprompt created by ChatPromptTemplate',prompt_template)

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

model = ChatOpenAI(model="gpt-4o-mini")

# def generate_llm_response(retrieved_documents, user_query):
#     ## making chain of prompt and llm
#     Chainof_Documents = create_stuff_documents_chain(llm=model,prompt=prompt_template)
#     print("\nChainof_Documents",Chainof_Documents)
#     retriever = vector_store.as_retriever()
#     print("\nretrievers -->>",retriever)
#     retrieval_chain =  create_retrieval_chain(retriever,Chainof_Documents)
#     # query = 'who is Admiral Winters?'
#     result = retrieval_chain.invoke({"input": user_query,"context":retrieved_documents})
#     # result = retrieval_chain.invoke({"input": user_query,"context":retrieved_documents})
#     print("\n'context'",result['context'])
#     print("\n'input'",result['input'])
#     print("\n'answer'",result['answer'])
#     # print("\nresult",result)
#     return result['answer']   

def generate_llm_response(retrieved_documents, user_query):
    ## making chain of prompt and llm
    prompt_template = ChatPromptTemplate.from_template("""You are an AI healthcare assistant. Use the information provided in the context to answer the user’s question thoroughly and accurately. Approach the answer step-by-step to ensure completeness and clarity.
                                                   
    Context Information:
    {context}

    User Question:
    {input}

    Please provide a detailed and insightful response:""")
    print('\nprompt created by ChatPromptTemplate',prompt_template)

    Chainof_Documents = create_stuff_documents_chain(llm=model,prompt=prompt_template)
    print("\nChainof_Documents",Chainof_Documents)
    retriever = vector_store.as_retriever()
    print("\nretrievers -->>",retriever)
    retrieval_chain =  create_retrieval_chain(retriever,Chainof_Documents)
    # query = 'who is Admiral Winters?'
    try:
        result = retrieval_chain.invoke({"input": user_query, "context": retrieved_documents})
        print("\nContext provided:", retrieved_documents)
        print("\nInput query:", result['input'])
        print("\nGenerated answer:", result['answer'])
        return result['answer']
    except Exception as e:
        print(f"Error during retrieval or response generation: {e}")
        return "An error occurred while generating the response."

# # Main execution
if __name__ == "__main__":
    # file_path = 'healthcare_dataset.csv'
    documents = read_and_load_csv(file_path)
    # print(documents)
    # Sort documents by billing amount (or any other desired field)
    sorted_documents = sort_documents(documents, sort_key="name")
    
    # Upsert sorted documents to Pinecone
    ingest_data_to_pinecone(sorted_documents)


    user_query="what is medial problem to KAREn johnsoN?"
    user_query1="how many paetient have bllod group a-"
    results = retrieve_data_from_pinecone(user_query1, top_k=5)
    # print('xxxxxx',results)
    generate_llm_response(results,user_query1)





# from langchain_openai import ChatOpenAI
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain
# # import openai
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain import hub



# template = "Given the context below, answer the question.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\nYOUR ANSWER:"
# prompt = ChatPromptTemplate.from_template(template)
# llm = ChatOpenAI()

# Initialize the ChatOpenAI model
# llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")

# def generate_response(matches, question):
#     # Initialize an empty list to hold context strings
#     contexts = []
    
#     # Extract relevant information from matches
#     for match in matches['matches']:
#         metadata = match['metadata']
#         # Construct a context string for each match
#         context = (
#             f"Name: {metadata['name']}, "
#             f"Age: {metadata['age']}, "
#             f"Admission Date: {metadata['admission_date']}, "
#             f"Medical Condition: {metadata['medical_condition']}, "
#             f"Billing Amount: ${metadata['billing_amount']:.2f}"
#         )
#         contexts.append(context)

#     # Join all context strings into one single string
#     context_string = "\n".join(contexts)
#     print('context_string\n',context_string)
#     # return context_string

#     # # Create messages for the chat model
#     messages = [
#         {"role": "system", "content": "You are an assistant that provides medical information."},
#         # {"role": "system", "content": context_string},
#         # {"role": "user", "content": context_string},
#         {"role": "user", "content": question}
#     ]

#     print(messages)
#     # Invoke the LLM with the formatted messages
#     response = llm.invoke(messages)
#     # response = openai.ChatCompletion.create(
#     #     model="gpt-3.5-turbo",
#     #     messages=messages,
#     #     temperature=0.7
#     # )
#     return response['choices'][0]['message']['content']
    # return response.content if hasattr(response, 'content') else response
   

    # # Create a chain for using the LLM with retrieved context
    # chain = llm | prompt
    # chain.invoke()
    # # chain = LLMChain(llm=llm, prompt=prompt)

    # # Define a function to process a user query
    # def answer_query(query):
    #     # Retrieve relevant records from Pinecone
    #     relevant_docs = retrieve_query(query=query)
        
    #     # Combine relevant records into a single context
    #     context = "\n".join([doc.page_content for doc in relevant_docs])
        
    #     # Run the LLM chain with context and query
    #     result = chain.invoke({"context": context})
    #     return result

    # # Example usage
    # # query = "What are the available services in cardiology?"
    # output = answer_query(query)
    # print(output)
    
  
# # ==================================
# from langchain_core.prompts import ChatPromptTemplate
# ## just search about hub is for ready made template

# prompt_template = ChatPromptTemplate.from_template("""
# Answer the following question based only on the provided context.
# Think step by step before providing a detailed answer.
# {context}
# Question:{input}
# """)
# print('\nprompt created by ChatPromptTemplate',prompt_template)

# from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAI 
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain

# model = ChatOpenAI(model="gpt-4o-mini")
# def generate_llm_response(retrieved_documents, user_query):
#     ## making chain of prompt and llm
#     Chainof_Documents = create_stuff_documents_chain(llm=model,prompt=prompt_template)
#     print("\nChainof_Documents",Chainof_Documents)
#     retriever = vector_store.as_retriever()
#     print("\nretrievers -->>",retriever)
#     retrieval_chain =  create_retrieval_chain(retriever,Chainof_Documents)
#     # query = 'who is Admiral Winters?'
#     result = retrieval_chain.invoke({"input": user_query})
#     print("\nresult['answer']",result['answer'])
#     # print("\nresult",result)
#     return result['answer']
# ==================================
# ==================================
# def generate_llm_response(retrieved_documents, user_query):
#     # Ensure the retrieved_documents are in the correct format
#     # Concatenate document content for the context
#     # context = "\n".join([doc["metadata"]["text"] for doc in retrieved_documents])
#     context = retrieved_documents
#     user_query=user_query
#     # Define a system prompt template to guide the LLM
#     system_template = f"Based on the following context, answer the question:\n\nContext:\n{context}"
#     user_template = "{text}"

#     # Define the prompt structure using ChatPromptTemplate
#     prompt_template = ChatPromptTemplate.from_messages([
#         ("system", system_template),
#         ("user", user_template)
#     ])

#     # Initialize the chain with the LLM and prompt template
#     chain = LLMChain(llm=llm, prompt=prompt_template)

#     # Invoke the chain with the user query
#     response = chain.invoke({"text": user_query})

#     # Print or return the generated response
#     print(response)
#     return response
# ==================================

# def generate_llm_response(retrieved_documents):
#     # Generate a response from the LLM
#     # user_query = "Find patients with cancer admitted in 2024"
#     # query = "Summarize the information for patients diagnosed with cancer, focusing on their age and billing amount."
#     print('generate_llm_response retrieved_documents',retrieved_documents)
#     # return
#     # Construct the prompt
#     # prompt = f""" {query}    {retrieved_documents}    """
#     # Step 3: Generate response using the language model
#     # prompt = f"Based on the following context, answer the question: '{user_query}'\n\nContext:\n{retrieved_documents}"
#     # generated_response = llm.generate(prompt)
#     # print('generate_llm_response',generated_response)
#     # # Get the response
#     # response = llm.generate(prompt)
#     # print(response)
#     context= retrieved_documents
#     # context = "\n".join([match["metadata"] for match in retrieved_documents["matches"]])
#     # prompt = ChatPromptTemplate.from_messages(
#     #     [("system", "Based on the following context, answer the question:\n\nContext:\n{context}")]
#     #     # [("system", "What are everyone's favorite colors:\n\n{context}")]
#     # )
#     # user_query = "Find patients with suger admitted in 2022"
#     system_template = {f"Based on the following context, answer the question:\n\nContext:\n{context}"}
#     prompt_template = ChatPromptTemplate.from_messages(
#     [("system", system_template), ("user", "{text}")]
#     )
#     llm = ChatOpenAI(model="gpt-3.5-turbo")
#     chain = create_stuff_documents_chain(llm, prompt_template)

#     # docs = [
#     #     Document(page_content="Jesse loves red but not yellow"),
#     #     Document(page_content = "Jamal loves green but not as much as he loves orange")
#     # ]
#     # chain.invoke({"context": docs})

#     # chain.invoke({"context": retrieved_documents})
#     chain = prompt_template | llm 
#     x= chain.invoke({ "text": user_query})
#     print(x)

# ==================================



# # Usage example
# if __name__ == "__main__":
#     # user_query = "Find patients with blood cancer admitted in 2020 and what is name of patients"  # Example user query
#     # metadata_filter = {"medical_condition": "Cancer", "admission_date": {"$gte": "2024-01-01"}}  # Optional filter
#     # results = retrieve_data_from_pinecone(user_query, top_k=5, metadata_filter=metadata_filter)
#     user_query="how many patients of the diabetis and give the cure of it?"
#     results = retrieve_data_from_pinecone(user_query, top_k=5)
#     generate_llm_response(results,user_query)
#     # generate_response(results,user_query)
#     # generate_llm_response(results, user_query)
#     # Display results
#     # for result in results:
#     #     print(f"ID: {result['id']}, Score: {result['score']}, Metadata: {result['metadata']}")