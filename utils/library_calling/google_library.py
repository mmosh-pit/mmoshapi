
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_pinecone import PineconeVectorStore
from utils.variable_constant.vertex_google import (project_id  ,location , model_id) 
import os
# from google.oauth2 import service_account

# credentials = service_account.Credentials.from_service_account_file('D:\All Desktop\iGenX\mmoshapi\service_account.json')


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001" , google_api_key=os.getenv('GENAI_API_KEY'))

# Pinecone vector store for context retrieval
pinecone_index_name = os.getenv('PINECONE_INDEX')
pinecone_store = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)