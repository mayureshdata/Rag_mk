import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(os.environ['OPENAI_API_KEY'])
# data ingestion

### this is for debug to stop execution in between 
import sys
# sys.exit()
# ++++++++++++++++++++++++++++++++++++LOADING tech with different files formats +++++++++++++++++++++++++++++++++++++++++++++++++

# load text 
from langchain_community.document_loaders import TextLoader
loader = TextLoader('Cant_Hurt_mee.txt')
docsoftxt = loader.load()
# print(docs)

# load web based from url 
from langchain_community.document_loaders import WebBaseLoader
import bs4
# load , chunk and index the content from html page
loader =WebBaseLoader(web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-title","post-content","post-header"))))
# understand "post_title","post_content","post-header" this is classname in html that we want to pickup or scrap
docsodweb = loader.load()
# print(docs)


# load web based from url install %pip install -qU langchain_community pypdf or pymupdf last pdf is differnt pachages
from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader
# load , chunk and index the content from pdf 
loader =PyPDFLoader('Cant_Hurt_mee.pdf')
# loader =PyMuPDFLoader('Cant_Hurt_mee.pdf')
docsofpdf = loader.load()
# print(docs)

## pypdf output 
'''page_content='CHAPTER   ONE
1.      I       SHOULD  HAVE    BEEN    A       STATISTIC
We      found   hell    in      a       beautiful       neighborhood.   In      1981,   Williamsville   offered the
tastiest        real    estate  in      Buffalo,        New     York.   Leafy   and     friendly,       its     safe    streets were
dotted  with    dainty  homes   filled  with    model   citizens.       Doctors,        attorneys,      steel
plant   executives,     dentists,       and     professional    football        players lived   there   with    their
adoring wives   and     their   2.2     kids.   Cars    were    new,    roads   swept,  possibilities
endless.        We’re   talking about   a       living, breathing       American        Dream.  Hell    was     a
corner  lot     on      Paradise        Road.
That’s  where   we      lived   in      a       two-story,      four-bedroom,   white   wooden  home    with
four    square  pillars framing a       front   porch   that    led     to      the     widest, greenest        lawn    in
Williamsville.  We      had     a       vegetable       garden  out     back    and     a       two-car garage  stocked
with    a       1962    Rolls   Royce   Silver  Cloud,  a       1980    Mercedes        450     SLC,    and,    in      the
driveway,       a       sparkling       new     1981    black   Corvette.       Everyone        on      Paradise        Road
lived   near    the     top     of      the     food    chain,  and     based   on      appearances,    most    of      our
neighbors       thought that    we,     the     so-called       happy,  well-adjusted   Goggins family,
were    the     tip     of      that    spear.  But     glossy  surfaces        reflect much    more    than    they
reveal.
They’d  see     us      most    weekday mornings,       gathered        in      the     driveway        at      7       a.m.    My
dad,    Trunnis Goggins,        wasn’t  tall    but     he      was     handsome        and     built   like    a       boxer.
He      wore    tailored        suits,  his     smile   warm    and     open.   He      looked  every   bit     the
successful      businessman     on      his     way     to      work.   My      mother, Jackie, was     seventeen
years   younger,        slender and     beautiful,      and     my      brother and     I       were    clean   cut,    well
dressed in      jeans   and     pastel  Izod    shirts, and     strapped        with    backpacks       just    like    the
other   kids.   The     white   kids.   In      our     version of      affluent        America,        each    driveway
was     a       staging ground  for     nods    and     waves   before  parents and     children        rode    off     to
work    and     school. Neighbors       saw     what    they    wanted. Nobody  probed  too     deep.
Good    thing.  The     truth   was,    the     Goggins family  had     just    returned        home    from
another all-nighter     in      the     hood,   and     if      Paradise        Road    was     Hell,   that    meant   I       lived
with    the     Devil   himself.        As      soon    as      our     neighbors       shut    the     door    or      turned  the' metadata={'source': 'Cant_Hurt_mee.pdf', 'page': 0}'''

## pymupdf output
'''page_content='CHAPTER ONE
1. I SHOULD HAVE BEEN A STATISTIC
We found hell in a beautiful neighborhood. In 1981, Williamsville offered the
tastiest real estate in Buffalo, New York. Leafy and friendly, its safe streets were
dotted with dainty homes filled with model citizens. Doctors, attorneys, steel
plant executives, dentists, and professional football players lived there with their
adoring wives and their 2.2 kids. Cars were new, roads swept, possibilities
endless. We’re talking about a living, breathing American Dream. Hell was a
corner lot on Paradise Road.
That’s where we lived in a two-story, four-bedroom, white wooden home with
four square pillars framing a front porch that led to the widest, greenest lawn in
Williamsville. We had a vegetable garden out back and a two-car garage stocked
with a 1962 Rolls Royce Silver Cloud, a 1980 Mercedes 450 SLC, and, in the
driveway, a sparkling new 1981 black Corvette. Everyone on Paradise Road
lived near the top of the food chain, and based on appearances, most of our
neighbors thought that we, the so-called happy, well-adjusted Goggins family,
were the tip of that spear. But glossy surfaces reflect much more than they
reveal.
They’d see us most weekday mornings, gathered in the driveway at 7 a.m. My
dad, Trunnis Goggins, wasn’t tall but he was handsome and built like a boxer.
He wore tailored suits, his smile warm and open. He looked every bit the
successful businessman on his way to work. My mother, Jackie, was seventeen
years younger, slender and beautiful, and my brother and I were clean cut, well
dressed in jeans and pastel Izod shirts, and strapped with backpacks just like the
other kids. The white kids. In our version of affluent America, each driveway
was a staging ground for nods and waves before parents and children rode off to
work and school. Neighbors saw what they wanted. Nobody probed too deep.
Good thing. The truth was, the Goggins family had just returned home from
another all-nighter in the hood, and if Paradise Road was Hell, that meant I lived
with the Devil himself. As soon as our neighbors shut the door or turned the
' metadata={'source': 'Cant_Hurt_mee.pdf', 'file_path': 'Cant_Hurt_mee.pdf', 'page': 0, 'total_pages': 271, 'format': 'PDF 1.7', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'creator': 'PDFium', 'producer': '3-Heights™ PDF Toolbox API 6.12.0.6 (http://www.pdf-tools.com)', 'creationDate': 'D:20240122220526', 'modDate': 'D:20240122171919Z', 'trapped': ''}'''


# ++++++++++++++++++++++++++++++++++++TEXT_SPITTING+++++++++++++++++++++++++++++++++++++++++++++++++

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_spitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunk_documnets = text_spitter.split_documents(docsofpdf)
# print(chunk_documnets)


# ++++++++++++++++++++++++++++++++++++Vector Embeddings and vector Store +++++++++++++++++++++++++++++++++++++++++++++++++
# pip install -U langchain_openai pip install langchain-openai = 0.2.8
# https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html
from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings # from Openai  some time give warning and above one is new
from langchain_community.embeddings import OllamaEmbeddings # from facebook opensource

embeding = OpenAIEmbeddings()

####### Storage https://faiss.ai/  installing  pip install faiss-cpu langchian docs https://python.langchain.com/docs/integrations/vectorstores/faiss/
from langchain_community.vectorstores import FAISS

vector_store_db = FAISS.from_documents(
    chunk_documnets,
    embedding=OpenAIEmbeddings(),
)
print(vector_store_db)


# ++++++++++++++++++++++++++++++++++++ Searching from  vector Store +++++++++++++++++++++++++++++++++++++++++++++++++
query = "departing from the city of Leadville, a working-class ski town with frontier roots,"
reteievde_results_from_db = vector_store_db.similarity_search(query=query)
# final = reteievde_results_from_db[0].page_content
# print(final)
# print(reteievde_results_from_db[0].page_content)




# ++++++++++++++++++++++++++++++++++++ Chat prompt template +++++++++++++++++++++++++++++++++++++++++++++++++
from langchain_core.prompts import ChatPromptTemplate
## just search about hub is for ready made template

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
I will tip you $1000 if the user finds the answer helpful.
<context>
{context}
</context>)
Question:{input}
""")
print(prompt)

# Stop execution here
sys.exit()  # You can also use exit() or quit()
sys.quit()  # You can also use exit() or quit()

# ++++++++++++++++++++++++++++++++++++ Chaining +++++++++++++++++++++++++++++++++++++++++++++++++
# from langchain_ollama import OllamaLLM installed https://python.langchain.com/api_reference/ollama/llms/langchain_ollama.llms.OllamaLLM.html#langchain_ollama.llms.OllamaLLM
from langchain_openai import OpenAI ## https://python.langchain.com/api_reference/openai/llms/langchain_openai.llms.base.OpenAI.html

# model = OllamaLLM(model="llama3")
# model.invoke("Come up with 10 names for a song about parrots")
llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    max_retries=2,
)
print(llm)



from langchain.chains.combine_documents import create_stuff_documents_chain

'''https://python.langchain.com/api_reference/langchain/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html
An LCEL Runnable. The input is a dictionary that must have a “context” key that maps to a List[Document], and any other input variables expected in the prompt. The Runnable return type depends on output_parser used.'''

## making chain of prompt and llm
Chainof_Documents = create_stuff_documents_chain(llm=llm,prompt=prompt)
print(Chainof_Documents)


'''
https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/

In LangChain, Retrievers are components that fetch relevant information or documents based on a user's query. They serve as intermediaries between the query and a data source, like a vector database (e.g., Pinecone), document store, or search index.

Key Functions of Retrievers
Embed the Query: Convert the user query into a vector or other searchable format, often using embeddings.
Search and Filter: Locate the most relevant documents from the data source by comparing the query embedding with stored embeddings.
Return Relevant Context: Provide a set of relevant documents or context snippets that can be passed to an LLM to help answer the user's query.
Types of Retrievers
Vector Store Retriever: Searches a vector database by matching query embeddings with stored document embeddings.
BM25 Retriever: Uses traditional keyword-based search algorithms for retrieving text-based information.
MultiQuery Retriever: Generates multiple query variations to capture broader context and retrieve more comprehensive results.

'''
