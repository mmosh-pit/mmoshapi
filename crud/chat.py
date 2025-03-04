
from utils.library_calling.google_library import (embeddings , pinecone_store)
from utils.library_calling.google_library import (HumanMessage , SystemMessage , AIMessage)
from typing import AsyncGenerator
from datetime import datetime, timezone
from utils.variable_constant.vertex_google import (project_id  ,location , model_id) 

import os
from models.database import db
from langsmith import traceable

@traceable(name="get_chat_history" , run_type="tool")
async def get_chat_history(username: str , n:int) -> list:
    """Fetch chat history for a user and convert to Langchain format."""
    collection = db.get_collection("users_chat_history")
    history = collection.find({"username": username}).sort("timestamp", -1).limit(n)
    
    # Return the messages as a list of dictionaries for Langchain compatibility
    chat_history = []
    for message in history:
        # Convert MongoDB message data into Langchain's message types (HumanMessage, AIMessage)
        if message["role"] == "user":
            chat_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "model":
            chat_history.append(AIMessage(content=message["content"]))
        elif message["role"] == "system":
            chat_history.append(SystemMessage(content=message["content"]))
    # print(chat_history)
    return chat_history

@traceable(name="save_chat_message" , run_type="tool")
def save_chat_message(username: str, role: str, content: str):
    """Save a chat message to MongoDB."""
    collection = db.get_collection("users_chat_history")
    collection.insert_one({
        "username": username,
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc)
    })
    
@traceable(name="get_relevant_context" , run_type="tool")
async def get_relevant_context(prompt: str, namespaces: list[str], metafield: str,) -> str:
    """Retrieve relevant context from Pinecone using vector search and optionally include a system prompt."""
    try:
        # Add "MMOSH" to the namespaces
        search_namespaces = namespaces + ["MMOSH"]

        all_search_results = []  # Make sure to initialize your list

        for namespace in search_namespaces:
            # Search in Pinecone for the relevant documents with scores
            search_results = pinecone_store.similarity_search_with_score(
                prompt,
                namespace=namespace,
                filter={"custom_metadata": metafield} if metafield and namespace != "MMOSH" else None,
            )
            
            # Filter and add only those results with a score of at least 0.6
            for doc, score in search_results:
                if score >= 0.6:
                    all_search_results.append(doc)

        # Combine all relevant information
        relevant_info = []
        for doc in all_search_results:
            relevant_info.append(doc.page_content.strip())

        return " ".join(relevant_info)


    except Exception as e:
        print(str(e))
        print(f"Error retrieving context: {str(e)}")
        return ""


