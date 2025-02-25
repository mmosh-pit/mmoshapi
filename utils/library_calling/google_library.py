from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_pinecone import PineconeVectorStore
from utils.variable_constant.vertex_google import (project_id  ,location , model_id) 
import os
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file('service_account.json')
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001" , google_api_key=os.getenv('GOOGLE_API_KEY') , credentials=credentials)
google_genai_model = GoogleGenerativeAI(
    model=model_id,  # Specify the appropriate Gemini model (e.g., text-bison)
    temperature=0,
    top_p=0.8,
    top_k=40,
    max_output_tokens=2048,
    project_id=project_id,  # Set your Google Cloud Project ID
    location=location  # Set your Google Cloud Region
)
# Pinecone vector store for context retrieval
pinecone_index_name = os.getenv('PINECONE_INDEX')
pinecone_store = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)