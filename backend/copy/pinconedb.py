import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.llms import OpenAI
import time

load_dotenv()
file_path = "data/Cant_Hurt_mee.pdf"
# file_path = "data/lllm.pdf"
API_KEY = os.getenv("OPENAI_API_KEY")
Pinecone_API_KEY = os.getenv("Pinecone_API_KEY")

# loader = PyPDFLoader(file_path)
# docs = loader.load()
# print(len(docs))

## Read all text from documents
def Read_doc(directory):
    file_loader = PyPDFLoader(directory)
    docs = file_loader.load()
    return docs

docs_readed_from_directory = Read_doc(file_path)
print(f"Number of documents loaded(number od pages in pdf): {len(docs_readed_from_directory)}")
# print(docs_readed_from_directory)

## text chunlikng of readed file 
def text_chunking(docs,chunk_size = 800,chunk_overlap=50):
    all_chunks_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks = all_chunks_text_splitter.split_documents(docs)
    print(f"Number of text chunks created: {len(chunks)}")
    return chunks

documents = text_chunking(docs=docs_readed_from_directory,chunk_size=800,chunk_overlap=50)
# print(len(documents))
# print(documents)


## text chunked to emmebdding
embeddings = OpenAIEmbeddings() 
# vectors = embeddings.embed_query('how are you?')  ## testing amy text and there vector
# print(vectors)
# print(len(vectors))


## PineCone datails and setup 
pc = Pinecone(Pinecone_API_KEY)
# database collection name in vector we say index
index_name = "imaxx"  # change if desired

#  Note THIS IS DEPRICATED
# Pinecone.init(api_key=Pinecone_API_KEY,host="https://imaxx-6gxod33.svc.aped-4627-b74a.pinecone.io",index_name=index_name)
# existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
# print(existing_indexes)

# import time
# if index_name not in existing_indexes:
#     pc.create_index(
#         name=index_name,
#         dimension=306,
#         # dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
#     while not pc.describe_index(index_name).status["ready"]:
#         time.sleep(1)

# index = pc.Index(index_name)
# vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())
# print(vector_store)


## adding local documnets in vectordb in imaxx index in pinecone
index = pc.Index(index_name)
print("details of created index in pinecone",index)
vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())

# vector_store.add_documents(documents=docs_readed_from_directory)
print(f"Documents added to Pinecone index '{index_name}'.")
## this is retreive from pincoden db Function to retrieve query from Pinecone
def retreive_query(query,k=2):
    matching_results = vector_store.similarity_search(query=query,k=k)
    return matching_results

# querytosearch= "what is NLP?"

# output =retreive_query(querytosearch)
# print(output[0])
# for res in output:
#     print(res)


from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
# from langchain.chains import RetrievalQA


# https://python.langchain.com/api_reference/langchain/hub.html#module-langchain.hub
# https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval.create_retrieval_chain.html#create-retrieval-chain
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever = vector_store.as_retriever()

# print("retriever information",retriever)

llm = ChatOpenAI(
    openai_api_key=API_KEY,
    # model_name='gpt-4o-mini',
    model_name='gpt-3.5-turbo',
    # model_name='gpt-4',
    temperature=0.2,
)

# https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval.create_retrieval_chain.html#create-retrieval-chain
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
# print('information about retrieval_chain ',retrieval_chain)

query_by_user = "what the book teach us?"

# https://python.langchain.com/docs/versions/migrating_chains/retrieval_qa/
answer_with_knowledge = retrieval_chain.invoke({"input": query_by_user})

print("\nAnswer with knowledge:\n\n", answer_with_knowledge['answer'])
print("\nContext used:\n", answer_with_knowledge['context'])
print("\n")
time.sleep(2)

# from langchain.llms import OpenAI

# llm = OpenAI(model="gpt-4")  # Use a model suitable for summarization or answering

# # Combine query and results for enhanced response
# context = "\n\n".join([result['text'] for result in output])
# final_response = llm.generate(f"{querytosearch}\nContext:\n{context}")
# print("LLM Response:")
# print(final_response)

# ## createing chains for better out put from llm 
# from langchain_openai import ChatOpenAI
# # llm = ChatOpenAI(model="gpt-4o-mini")

# from langchain_core.prompts import ChatPromptTemplate

# # Setup LangChain to create a conversational response
# llm = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY)
# def get_answers(query):
#     # uses the above model (llm) and give the answers which is suppose to do that
#     # system_template = "Translate the following into {language}:"
#     data = retreive_query(query=query)
#     system_template = {f"take this retreved qurey from vectordb and analys answers from it and give apropriate answers {data}"}
#     prompt_template = ChatPromptTemplate.from_messages(
#     [("system", system_template), ("user", "{text}")]
#     )
    
#     #2) method 2
#     from langchain_core.output_parsers import StrOutputParser
#     parser = StrOutputParser()
#     chain = prompt_template | llm | parser
#     x= chain.invoke({ "text": query})
#     print(x)


# get_answers("what are limits of Transfer Learning ?")










# ==================================

# import os
# import time
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain
# import pinecone

# # Load environment variables
# load_dotenv()
# file_path = "data/lllm.pdf"
# API_KEY = os.getenv("OPENAI_API_KEY")
# Pinecone_API_KEY = os.getenv("Pinecone_API_KEY")

# # Load and read documents from the PDF file
# def read_doc(directory):
#     file_loader = PyPDFLoader(directory)
#     docs = file_loader.load()
#     return docs

# docs_readed_from_directory = read_doc(file_path)

# # Split text into chunks for better embeddings
# def text_chunking(docs, chunk_size=800, chunk_overlap=50):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     chunks = text_splitter.split_documents(docs)
#     return chunks

# documents = text_chunking(docs_readed_from_directory)

# # Initialize embeddings and Pinecone
# embeddings = OpenAIEmbeddings(api_key=API_KEY)

# pinecone.init(api_key=Pinecone_API_KEY, environment="us-east-1-gcp")
# index_name = "imaxx"

# # Create or connect to the Pinecone index
# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(index_name, dimension=1536, metric="cosine")

# index = pinecone.Index(index_name)
# vector_store = Pinecone(index=index, embedding=embeddings)

# # Add documents to the Pinecone vector store
# vector_store.add_documents(documents)

# # Function to retrieve query from Pinecone
# def retrieve_query(query, k=2):
#     matching_results = vector_store.similarity_search(query=query, k=k)
#     return matching_results

# # Setup LangChain to create a conversational response
# llm = ChatOpenAI(model="gpt-4", api_key=API_KEY)
# prompt = ChatPromptTemplate.from_template("Summarize this content: {context}")

# # Create a chain for better output from LLM
# chain = LLMChain(llm=llm, prompt=prompt)

# # Function to run query and use LLM for final answer
# def answer_query(query):
#     # Retrieve relevant documents from Pinecone
#     relevant_docs = retrieve_query(query=query)
    
#     # Combine retrieved docs into a single context
#     context = " ".join([doc.page_content for doc in relevant_docs])
    
#     # Run through the LLM chain
#     result = chain.invoke({"context": context})
#     return result

# # Example query
# query = "What is NLP?"
# output = answer_query(query)
# print(output)
