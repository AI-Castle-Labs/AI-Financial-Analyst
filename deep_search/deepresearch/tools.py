
from serpapi import GoogleSearch
from langchain.tools import Tool
#from general_tools.tools import chunk_long_words_in_text
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from langchain.embeddings import OpenAIEmbeddings
import uuid
import requests
from bs4 import BeautifulSoup
from langchain.utilities import SerpAPIWrapper
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams
from langchain.document_loaders import PyPDFLoader
from openai import OpenAI

#brave search api
"""
1)Planner Agent
2)WebResearch Agent
3)
4)
5)Syntesizer agent
6)Memory Agent



"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

api_key = "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"
qdrant_url="https://6629bf04-24f3-4136-af1f-86fcacc5db74.eu-west-2-0.aws.cloud.qdrant.io:6333"
qdrant_api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.hlNhEkXzv5RrfhdaVKadeG1LUkyxP-lnLAwQ4yYEnM4"


def summarizer(conversation):
    """This tool acts a short-term-memory which can be used for storing session memory and user chats to have better output"""
    memory = {}
    

def scrapper(web_link):
    try:
        response = requests.get(web_link, headers = {'User-Agent':'Mozilla/5.0'})
        soup = BeautifulSoup(response.text,'html.parser')
        titles = soup.find_all('h2', class_='article-title')
        return soup
    except Exception as e:
        print(e)



def use_serp_tool(query):
    serp_api_wrapper = SerpAPIWrapper(serpapi_api_key="83ba35cb160d0fbc08b3972c94eeb6952054c7b57ffd77002c72c0813232daae")
    return serp_api_wrapper.run(query)

# Define Tool if not already defined or import it from the appropriate module
# from some_module import Tool

#pdf_path = r"C:\Users\ashle\OneDrive\Desktop\search\AshleynCastelino_Resume (3).pdf"
#loader = PyPDFLoader(pdf_path)
#documents = loader.load()
#print(documents)
#collection_name = "Financial_Documents"


def store_document():
    client = QdrantClient(
        url="https://6629bf04-24f3-4136-af1f-86fcacc5db74.eu-west-2-0.aws.cloud.qdrant.io:6333", 
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.hlNhEkXzv5RrfhdaVKadeG1LUkyxP-lnLAwQ4yYEnM4",
    )
    chunks = text_splitter.split_documents(documents)
    collection_name = "Financial_Documents"
    if collection_name not in client.get_collections().collections:
        client.recreate_collection(
            collection_name = collection_name,
            vectors_config=VectorParams(size = 1536, distance = Distance.COSINE)
        )
    client = OpenAI(api_key = api_key)
    pdf_path = r"C:\Users\ashle\OneDrive\Desktop\search\AshleynCastelino_Resume (3).pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    response = client.embeddings.create(
        input = documents,
        model="text-embedding-3-small"
        )
    
    #Embed/vectorize the document before sending it to upload
    upload = client.upload_collection(
            collection_name = collection_name,
            vectors = chunks
        )


def long_term_memory():
    client = QdrantClient(":memory:")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    qdrant_client = QdrantClient(
        url="https://6629bf04-24f3-4136-af1f-86fcacc5db74.eu-west-2-0.aws.cloud.qdrant.io:6333", 
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.hlNhEkXzv5RrfhdaVKadeG1LUkyxP-lnLAwQ4yYEnM4",
    )
    print(qdrant_client.get_collections())

def store_document(document =r"C:\Users\ashle\OneDrive\Desktop\search\S&P_500 research report.pdf" ):
   
    qdrant_client = QdrantClient(url=qdrant_url, api_key= qdrant_api_key)
    collection_name = "Financial Document"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    loader = PyPDFLoader(document)
    documents = loader.load()

    chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key = api_key)


    existing = qdrant_client.get_collections().collections
    if collection_name not in [col.name for col in existing]:
        qdrant_client.recreate_collection(
            collection_name = collection_name,
            vectors_config = VectorParams(size = 1536, distance = Distance.EUCLID)
        )
    
    vectorstore = Qdrant.from_documents(chunks, embeddings, collection_name=collection_name, 
                                        url=qdrant_url, api_key=qdrant_api_key,distance_func=Distance.EUCLID )
    
    print("Documents stored in Qdrant successfully.",vectorstore)
    return vectorstore    

def retrieve_document(query):
    qdrant_client = QdrantClient(url = qdrant_url, api_key = qdrant_api_key)
    collection_name = "Financial Document"

    embeddings = OpenAIEmbeddings(openai_api_key = api_key)
    
    client = QdrantClient(url=qdrant_url, api_key= qdrant_api_key)
    qdrant =  Qdrant(client, collection_name, embeddings)
    results = qdrant.similarity_search(query, k = 5)
    return results


available_tools = [
    Tool(
        name="web_search",
        func=use_serp_tool,
        description="Performs a Google search when confidence in internal knowledge is low."
    ),
    Tool(
        name="long_term_memory",
        func=long_term_memory,
        description="Accesses or stores data in Qdrant vector database for long-term memory."
    ),
    Tool(
        name = "Load Docuemnt",
        func = store_document,
        description = "Tool to add documents to the database"
    )
    
]

