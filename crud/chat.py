from google.generativeai.protos import (
    Content,
    Part,
)
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from datetime import datetime, timezone
import os
from models.database import db


async def get_chat_history(username: str) -> list[Content]:
    """Fetch chat history for a user and convert to Gen AI Content format."""
    collection = db.get_collection("users_chat_history")
    history = collection.find({"username": username}).sort("timestamp", 1)
    chat_history = []
    for message in history:
        chat_history.append(Content(
            role=message["role"],
            parts=[Part(text=message["content"])]
        ))
    
    return chat_history

def save_chat_message(username: str, role: str, content: str):
    """Save a chat message to MongoDB."""
    collection = db.get_collection("users_chat_history")
    collection.insert_one({
        "username": username,
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc)
    })

async def get_relevant_context(prompt: str, namespaces: list[str], metafield: str,) -> str:
    """Retrieve relevant context from Pinecone using vector search and optionally include a system prompt."""
    try:
        # Initialize embeddings
        embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
        
        # Add "MMOSH" to the namespaces
        search_namespaces = namespaces + ["MMOSH"]
        
        all_search_results = []
        
        
        for namespace in search_namespaces:
            vectorstore = PineconeVectorStore(
                index_name=os.getenv('PINECONE_INDEX'),
                embedding=embeddings,
                namespace=namespace
            )
            
            # Add metadata filtering if metafield is provided
            if metafield and namespace != "MMOSH":
                search_results = vectorstore.similarity_search(
                    prompt,
                    filter={"custom_metadata": metafield}
                )
            else:
                search_results = vectorstore.similarity_search(prompt)
            
            all_search_results.extend(search_results)

        # Combine all relevant information
        relevant_info = []
        for doc in all_search_results:
            relevant_info.append(doc.page_content.strip())
        return " ".join(relevant_info)
        
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        return ""

