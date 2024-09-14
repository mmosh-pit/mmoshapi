import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Form
from fastapi.responses import StreamingResponse
import asyncio
from typing import AsyncGenerator
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime, timezone
import vertexai
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
)
import certifi
import requests

from langchain_pinecone import PineconeVectorStore

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community.google_speech_to_text import SpeechToTextLoader
import os
from starlette.middleware.base import BaseHTTPMiddleware

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_vertexai import VertexAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


load_dotenv()

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

ca_cert_path = certifi.where()

PROJECT_ID = os.getenv('VERTEX_PROJECT_ID')

index_name = os.getenv('PINECONE_INDEX')

class LargeRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == 'POST':
            MAX_BODY_SIZE = 100 * 1024 * 1024  # 100 MB
            content_length = int(request.headers.get('Content-Length', 0))
            if content_length > MAX_BODY_SIZE:
                return JSONResponse(status_code=413, content={"message": "Request payload too large"})
        return await call_next(request)

app = FastAPI()

# Add the custom middleware
app.add_middleware(LargeRequestMiddleware)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI') + f'&tls=true&tlsCAFile={ca_cert_path}'
#MONGO_URI = os.getenv('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client.get_database("moral_panic_bot")

# Pydantic model for request validation
class GenerateRequest(BaseModel):
    username: str
    prompt: str

# Dependency to fetch chat history
async def get_chat_history(username: str):
    collection = db.get_collection("users_chat_history")
    history = collection.find({"username": username}).sort("timestamp", 1)  # Assuming you have a timestamp field for sorting
    chat_history = []

    for message in history:
        chat_history.append(Content(
            role=message["role"],
            parts=[Part.from_text(message["content"])]
        ))
        
    return chat_history

def save_chat_message(username: str, role: str, content: str):
    collection = db.get_collection("users_chat_history")
    collection.insert_one({
        "username": username,
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc)
    })

async def generate_function_call_stream(prompt: str, username: str, project_id: str, location: str, chat_history: list, namespaces: list, metafield: str) -> AsyncGenerator[str, None]:

    try:

        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        print(prompt)

        # Specify a function declaration and parameters for an API request
        get_command_func = FunctionDeclaration(
            name="get_command",
            description=(
                "Execute a specific command based on user's clear intention to take action. "
                "This function ONLY returns one of the following commands: '/start', '/earn', '/main', '/bags', '/settings', '/connect', '/status', '/join'. "
                "It should be called ONLY when the user explicitly expresses an intention to perform one of these actions. "
                "The function returns ONLY the command string, without any additional text."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["/start", "/earn", "/main", "/bags", "/settings", "/connect", "/status", "/join"],
                        "description": "The specific command to execute based on user's clear intention."
                    }
                },
                "required": ["command"],
            },
        )

        get_information_func = FunctionDeclaration(
            name="get_information",
            description=(
                "Retrieve relevant information from the database to answer user questions or respond to general chat messages. "
                "This function should be called for all user inputs that do not explicitly indicate an intention to execute a command. "
                "It retrieves information to help answer questions, provide explanations, or engage in general conversation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question or message to find relevant information for."
                    }
                },
                "required": ["query"],
            },
        )


        command_tool = Tool(
            function_declarations=[get_command_func, get_information_func],
        )

        system_instructions = (
            "System Instructions: You are an AI bot designed to assist users. Your primary functions are: "
            "\n1. **Command Execution:** When a user's query clearly indicates the intent to execute a specific command "
            "(only '/start', '/earn', '/main', '/bags', '/settings', '/connect', '/status', '/join'), "
            "respond by calling the get_command function with ONLY that command, without any additional text. "
            "Do not invent or use any commands other than these specified ones."
            "\n2. **Information Provision:** For ALL other queries, including questions, requests for information, "
            "or general chat messages that don't explicitly indicate command execution, call the get_information function. "
            "Use the information retrieved from the database to provide comprehensive and informative responses. "
            "Do not make up or add any information beyond what is provided by the get_information function."
            "\n\nYour goal is to be helpful and informative while strictly adhering to these guidelines. "
            "Always use one of these two functions for every user input, choosing based on the nature of the user's message."
        )


        user_prompt_content = Content(
            role="user",
            parts=[
                Part.from_text(system_instructions + " User prompt: " + prompt),
            ],
        )

        generation_config = GenerationConfig(
            temperature=0,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )

            # Safety config - set all categories to BLOCK_ONLY_HIGH
        safety_config = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]

        model = GenerativeModel(
            model_name="gemini-1.0-pro-001",
            generation_config=generation_config,
            safety_settings=safety_config,
            tools=[command_tool],
        )


        # Start a chat session
        chat = model.start_chat(history=chat_history if chat_history else None)

        response = chat.send_message(user_prompt_content)

        if not response.candidates:
            yield "I'm sorry, I couldn't generate a response. Please try again."
            return

        # Determine which function was called based on the response or handle text responses
        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call

            print(function_call)
            
            if function_call.name == "get_command":
                command_type = function_call.args["command"]
                #api_response = f"{{'command': '{command_type}'}}"
                yield command_type
                save_chat_message(username, "user", prompt)
                save_chat_message(username, "model", command_type)
            elif function_call.name == "get_information":
                embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
        
                # Add "MMOSH" to the namespaces
                search_namespaces = namespaces + ["MMOSH"]
                
                all_search_results = []
                for namespace in search_namespaces:
                    print(namespace)
                    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
                    
                    # Add metadata filtering if metafield is provided
                    if metafield and namespace != "MMOSH":
                        search_results = vectorstore.similarity_search(prompt, filter={"custom_metadata": metafield})
                    else:
                        search_results = vectorstore.similarity_search(prompt)
                    
                    all_search_results.extend(search_results)
                
                relevant_info = []
                for doc in all_search_results:
                    relevant_info.append(doc.page_content.strip())  # Remove leading/trailing whitespace

                print(relevant_info)

                api_response = {
                    "information": " ".join(relevant_info)  # Combine results into a single string
                }

                # Send the function response
                response = chat.send_message(
                    Part.from_function_response(
                        name=function_call.name,
                        response={
                            "content": api_response,
                        },
                    ),
                )

                # Stream the response
                for chunk in response.text.split():  # Simple word-by-word streaming
                    yield chunk + " "
                    await asyncio.sleep(0.05)  # Add a small delay between words

                save_chat_message(username, "user", prompt)
                save_chat_message(username, "model", response.text)

            else:
                yield "I'm sorry, I don't know how to process that request. Here's what I can do: /start, /earn, /main, /bags, /settings, /connect, /status, /join."
        else:
            response_text = response.candidates[0].content.parts[0].text
            if response_text:
                for chunk in response_text.split():
                    yield chunk + " "
                    await asyncio.sleep(0.05)
                save_chat_message(username, "user", prompt)
                save_chat_message(username, "model", response.text)
            else:
                yield "I'm sorry, I couldn't generate a response. Please try again."


        # Save the chat messages after streaming
        #save_chat_message(username, "user", prompt)
        #save_chat_message(username, "model", response.text)

    except Exception as e:
        yield f"An error occurred: {str(e)}"


@app.post("/generate_stream/")
async def get_generate_stream(request: Request):
    try:
        data = await request.json()
        username = data.get('username')
        prompt = data.get('prompt')
        namespaces = data.get('namespaces', [])  # Changed from 'coinsList' to 'namespaces'
        metafield = data.get('metafield', '')  # New parameter
        if not username or not prompt:
            raise HTTPException(status_code=400, detail="Username and prompt fields are required")
        chat_history = await get_chat_history(username)
        
        return StreamingResponse(
            generate_function_call_stream(prompt, username, PROJECT_ID, "us-central1", chat_history, namespaces, metafield),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 

def generate_function_call(prompt: str, username: str, project_id: str, location: str, chat_history: list, namespaces: list, metafield: str) -> str:

    try:

        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        print(prompt)

        # Specify a function declaration and parameters for an API request
        get_command_func = FunctionDeclaration(
            name="get_command",
            description=(
                "Execute a specific command based on user's clear intention to take action. "
                "This function ONLY returns one of the following commands: '/start', '/earn', '/main', '/bags', '/settings', '/connect', '/status', '/join'. "
                "It should be called ONLY when the user explicitly expresses an intention to perform one of these actions. "
                "The function returns ONLY the command string, without any additional text."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["/start", "/earn", "/main", "/bags", "/settings", "/connect", "/status", "/join"],
                        "description": "The specific command to execute based on user's clear intention."
                    }
                },
                "required": ["command"],
            },
        )

        get_information_func = FunctionDeclaration(
            name="get_information",
            description=(
                "Retrieve relevant information from the database to answer user questions or respond to general chat messages. "
                "This function should be called for all user inputs that do not explicitly indicate an intention to execute a command. "
                "It retrieves information to help answer questions, provide explanations, or engage in general conversation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question or message to find relevant information for."
                    }
                },
                "required": ["query"],
            },
        )


        command_tool = Tool(
            function_declarations=[get_command_func, get_information_func],
        )

        system_instructions = (
            "System Instructions: You are an AI bot designed to assist users. Your primary functions are: "
            "\n1. **Command Execution:** When a user's query clearly indicates the intent to execute a specific command "
            "(only '/start', '/earn', '/main', '/bags', '/settings', '/connect', '/status', '/join'), "
            "respond by calling the get_command function with ONLY that command, without any additional text. "
            "Do not invent or use any commands other than these specified ones."
            "\n2. **Information Provision:** For ALL other queries, including questions, requests for information, "
            "or general chat messages that don't explicitly indicate command execution, call the get_information function. "
            "Use the information retrieved from the database to provide comprehensive and informative responses. "
            "Do not make up or add any information beyond what is provided by the get_information function."
            "\n\nYour goal is to be helpful and informative while strictly adhering to these guidelines. "
            "Always use one of these two functions for every user input, choosing based on the nature of the user's message."
        )


        user_prompt_content = Content(
            role="user",
            parts=[
                Part.from_text(system_instructions + " User prompt: " + prompt),
            ],
        )

        generation_config = GenerationConfig(
            temperature=0,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )

            # Safety config - set all categories to BLOCK_ONLY_HIGH
        safety_config = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]

        model = GenerativeModel(
            model_name="gemini-1.0-pro-001",
            generation_config=generation_config,
            safety_settings=safety_config,
            tools=[command_tool],
        )



        # Start a chat session
        chat = model.start_chat(history=chat_history if chat_history else None)

        response = chat.send_message(user_prompt_content)

        if not response.candidates:
            return "I'm sorry, I couldn't generate a response. Please try again."

        # Determine which function was called based on the response or handle text responses
        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call

            print(function_call)
            
            if function_call.name == "get_command":
                command_type = function_call.args["command"]
                #api_response = f"{{'command': '{command_type}'}}"
                return command_type
            elif function_call.name == "get_information":
                embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
        
                # Add "MMOSH" to the namespaces
                search_namespaces = namespaces + ["MMOSH"]
                
                all_search_results = []
                for namespace in search_namespaces:
                    print(namespace)
                    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
                    
                    # Add metadata filtering if metafield is provided
                    if metafield and namespace != "MMOSH":
                        search_results = vectorstore.similarity_search(prompt, filter={"custom_metadata": metafield})
                    else:
                        search_results = vectorstore.similarity_search(prompt)
                    
                    all_search_results.extend(search_results)
                
                relevant_info = []
                for doc in all_search_results:
                    relevant_info.append(doc.page_content.strip())  # Remove leading/trailing whitespace

                api_response = {
                    "information": " ".join(relevant_info)  # Combine results into a single string
                }


            else:
                return "I'm sorry, I don't know how to process that request. Here's what I can do: /start, /earn, /main, /bags, /settings, /connect, /status, /join."
        else:
            response_text = response.candidates[0].content.parts[0].text
            print(response_text)
            if response_text:
                save_chat_message(username, "user", prompt)
                save_chat_message(username, "model", response.text)
                return response_text
            else:
                return "I'm sorry, I couldn't generate a response. Please try again."

        response = chat.send_message(
            Part.from_function_response(
                name=function_call.name,
                response={
                    "content": api_response,
                },
            ),
        )

        # Save the current prompt and response to MongoDB
        save_chat_message(username, "user", prompt)
        save_chat_message(username, "model", response.text)

        return response.text

    except Exception as e:
        return "I apologize, but I encountered an unexpected error. Please try again with different query."

@app.post("/generate/")
async def get_generate(request: Request):
    try:
        data = await request.json()
        username = data.get('username')
        prompt = data.get('prompt')
        namespaces = data.get('namespaces', [])  # Changed from 'coinsList' to 'namespaces'
        metafield = data.get('metafield', '')  # New parameter
        if not username or not prompt:
            raise HTTPException(status_code=400, detail="Username and prompt fields are required")
        chat_history = await get_chat_history(username)
        summary = generate_function_call(prompt, username, PROJECT_ID, "us-central1", chat_history, namespaces, metafield)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Create the documents folder if it does not exist
if not os.path.exists('documents'):
    os.makedirs('documents')

from google.cloud import speech, storage
import io
import wave
import aiofiles
import tempfile
import shutil
from datetime import timedelta
import wave
import uuid
from typing import List


@app.post("/upload")
async def upload_file(
    name: str = Form(...),
    metadata: str = Form(...),
    file: UploadFile = File(None),
    urls: List[str] = Form(...)
):
    try:
        all_documents = []

        if file and file.filename:
            # Handle audio file upload
            content = await file.read()
            content_type = file.content_type
            if 'audio/wav' not in content_type:
                return JSONResponse(status_code=400, content={"message": "Attached file must be a WAV audio file."})
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(content)
                temp_audio_path = temp_audio.name

            try:
                loader = SpeechToTextLoader(
                    project_id=os.getenv('VERTEX_PROJECT_ID'),
                    file_path=temp_audio_path,
                    location="us-central1",
                    recognizer_id="_",
                    config_mask=None,
                    is_long=False
                )
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as speech_error:
                print(f"Speech-to-Text error: {str(speech_error)}")
                return JSONResponse(status_code=500, content={"message": f"Speech-to-Text error: {str(speech_error)}"})
            finally:
                os.unlink(temp_audio_path)

        for url in urls:
            response = requests.get(url)
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type')
                
                if 'application/pdf' in content_type:
                    file_extension = 'pdf'
                elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
                    file_extension = 'docx'
                else:
                    return JSONResponse(status_code=400, content={"message": f"URL {url} must point to a PDF or DOCX file."})
                
                # Save the content to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                    temp_file.write(response.content)
                    temp_file_path = temp_file.name
                
                if file_extension == 'pdf':
                    loader = PyPDFLoader(temp_file_path)
                elif file_extension == 'docx':
                    loader = Docx2txtLoader(temp_file_path)
                
                documents = loader.load()
                all_documents.extend(documents)
                
                # Clean up the temporary file
                os.unlink(temp_file_path)
            else:
                return JSONResponse(status_code=402, content={"message": f"Failed to download file from {url}. Status code: {response.status_code}"})

        if not all_documents:
            return JSONResponse(status_code=400, content={"message": "No content found in any of the files"})

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        doc_splits = text_splitter.split_documents(all_documents)

        for doc in doc_splits:
            doc.metadata['custom_metadata'] = metadata
            # Convert timedelta objects to strings
            for key, value in doc.metadata.items():
                if isinstance(value, timedelta):
                    doc.metadata[key] = str(value)

        embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")

        PineconeVectorStore.from_documents(doc_splits, embeddings, index_name=index_name, namespace=name)
        
        return JSONResponse(status_code=200, content={"message": "Files uploaded successfully"})
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})


import random

@app.get("/fetch_namespaces")
async def fetch_namespaces():
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        # Initialize the index
        index = pc.Index(index_name)
        
        # Fetch index stats to get namespaces
        stats = index.describe_index_stats()
        namespaces = stats.namespaces

        # Initialize a list to store namespace data
        namespace_data = []

        # Process namespace information and fetch all unique metadata
        for namespace, ns_stats in namespaces.items():
            unique_metadata = set()
            if ns_stats.vector_count > 0:
                try:
                    # Query multiple vectors from the namespace
                    query_response = index.query(
                        vector=[random.random() for _ in range(768)],  # Random vector with 768 dimensions
                        top_k=min(ns_stats.vector_count, 100),  # Fetch up to 100 vectors or all if less
                        namespace=namespace,
                        include_metadata=True
                    )
                    for match in query_response.matches:
                        if match.metadata and 'custom_metadata' in match.metadata:
                            unique_metadata.add(match.metadata['custom_metadata'])
                except Exception as e:
                    print(f"Error fetching metadata for namespace {namespace}: {str(e)}")

            namespace_data.append({
                "namespace": namespace,
                "vector_count": ns_stats.vector_count,
                "unique_metadata": list(unique_metadata) if unique_metadata else ["No metadata available"]
            })

        return JSONResponse(status_code=200, content={"namespaces": namespace_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.delete("/delete_by_metadata")
async def delete_by_metadata(metadata: str):
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        # Initialize the index
        index = pc.Index(index_name)
        
        # Fetch index stats to get namespaces
        stats = index.describe_index_stats()
        namespaces = stats.namespaces

        total_deleted = 0
        debug_info = []

        for namespace in namespaces:
            try:
                # Query to find vectors with the specified metadata
                query_response = index.query(
                    vector=[0] * 768,  # Dummy vector
                    top_k=10000,  # Increase this if you have more vectors
                    namespace=namespace,
                    filter={"custom_metadata": {"$eq": metadata}},
                    include_metadata=True
                )
                
                matching_ids = [match.id for match in query_response.matches]
                matches_found = len(matching_ids)
                
                if matching_ids:
                    # Delete the matching vectors
                    delete_response = index.delete(ids=matching_ids, namespace=namespace)
                    
                    # Handle the case where delete_response is a dict
                    if isinstance(delete_response, dict):
                        deleted_count = delete_response.get('deleteCount', matches_found)
                    else:
                        deleted_count = getattr(delete_response, 'delete_count', matches_found)
                    
                    total_deleted += deleted_count
                    
                    debug_info.append(f"Namespace: {namespace}, Matches found: {matches_found}, Deleted: {deleted_count}")
                else:
                    debug_info.append(f"Namespace: {namespace}, No matches found")
                
                # Add information about the remaining metadata in this namespace
                post_delete_query = index.query(
                    vector=[0] * 768,
                    top_k=5,
                    namespace=namespace,
                    include_metadata=True
                )
                remaining_metadata = [match.metadata.get('custom_metadata') for match in post_delete_query.matches if match.metadata]
                debug_info.append(f"Remaining metadata samples in {namespace}: {remaining_metadata}")
            
            except Exception as e:
                debug_info.append(f"Error in namespace {namespace}: {str(e)}")

        if total_deleted > 0:
            return JSONResponse(status_code=200, content={
                "message": f"Deleted {total_deleted} vectors with metadata '{metadata}'",
                "debug_info": debug_info
            })
        else:
            return JSONResponse(status_code=404, content={
                "message": f"No vectors found or deleted with metadata '{metadata}'",
                "debug_info": debug_info
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete_namespace")
async def delete_namespace(namespace: str):
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        # Initialize the index
        index = pc.Index(index_name)
        
        # Fetch index stats to get namespace information
        stats = index.describe_index_stats()
        namespaces = stats.namespaces

        if namespace not in namespaces:
            return JSONResponse(status_code=404, content={
                "message": f"Namespace '{namespace}' not found"
            })

        # Get the vector count before deletion
        vector_count_before = namespaces[namespace].get('vector_count', 0)

        # Delete all vectors in the namespace
        try:
            delete_response = index.delete(delete_all=True, namespace=namespace)
        except Exception as delete_error:
            return JSONResponse(status_code=500, content={
                "message": f"Error during deletion: {str(delete_error)}",
                "vector_count_before": vector_count_before
            })

        # Get the vector count after deletion
        stats_after = index.describe_index_stats()
        vector_count_after = stats_after.namespaces.get(namespace, {}).get('vector_count', 0)

        vectors_deleted = vector_count_before - vector_count_after

        return JSONResponse(status_code=200, content={
            "message": f"Attempted to delete all vectors in namespace '{namespace}'",
            "vectors_deleted": vectors_deleted,
            "vector_count_before": vector_count_before,
            "vector_count_after": vector_count_after
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "message": f"An error occurred: {str(e)}"
        })



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
