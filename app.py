import uvicorn
from fastapi import FastAPI, HTTPException, Request, Form
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
from dotenv import load_dotenv
import os

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_vertexai import VertexAIEmbeddings

#from langchain_google_vertexai import VertexAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

load_dotenv()

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

ca_cert_path = certifi.where()

PROJECT_ID = os.getenv('VERTEX_PROJECT_ID')

index_name = os.getenv('PINECONE_INDEX')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
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

async def generate_function_call_stream(prompt: str, username: str, project_id: str, location: str, chat_history: list, coinsList: list) -> AsyncGenerator[str, None]:

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
                
                # Add "MMOSH" to the coinsList
                search_namespaces = coinsList + ["MMOSH"]
                
                all_search_results = []
                for namespace in search_namespaces:
                    print(namespace)
                    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
                    search_results = vectorstore.similarity_search(prompt)
                    all_search_results.extend(search_results)
                    
                relevant_info = []
                for doc in all_search_results:
                    relevant_info.append(doc.page_content.strip())  # Remove leading/trailing whitespace

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
        coinsList = data.get('coinsList', [])
        if not username or not prompt:
            raise HTTPException(status_code=400, detail="Username and prompt fields are required")
        chat_history = await get_chat_history(username)
        
        return StreamingResponse(
            generate_function_call_stream(prompt, username, PROJECT_ID, "us-central1", chat_history, coinsList),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 

def generate_function_call(prompt: str, username: str, project_id: str, location: str, chat_history: list, coinsList: list) -> str:

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
                
                # Add "MMOSH" to the coinsList
                search_namespaces = coinsList + ["MMOSH"]
                
                all_search_results = []
                for namespace in search_namespaces:
                    print(namespace)
                    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
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
        coinsList = data.get('coinsList', [])  # Default to empty list if not provided
        if not username or not prompt:
            raise HTTPException(status_code=400, detail="Username and prompt fields are required")
        chat_history = await get_chat_history(username)
        summary = generate_function_call(prompt, username, PROJECT_ID, "us-central1", chat_history, coinsList)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Create the documents folder if it does not exist
if not os.path.exists('documents'):
    os.makedirs('documents')

@app.post("/upload")
async def upload_file(name: str = Form(...), url: str = Form(...)):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type')

            print(content_type)
            
            if 'application/pdf' in content_type:
                file_extension = 'pdf'
            elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
                file_extension = 'docx'
            else:
                return JSONResponse(status_code=400, content={"message": "Unsupported file type"})
            
            file_path = os.path.join('documents', f"{name}.{file_extension}")
            with open(file_path, "wb") as buffer:
                buffer.write(response.content)
            
            if file_extension == 'pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == 'docx':
                loader = Docx2txtLoader(file_path)
            else:
                return JSONResponse(status_code=400, content={"message": "Unsupported file type"})

            pages = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=20,
                length_function=len,
                is_separator_regex=False,
            )
            doc_splits = text_splitter.split_documents(pages)

            embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")

            PineconeVectorStore.from_documents(doc_splits, embeddings, index_name=index_name, namespace=name)
            
            return JSONResponse(status_code=200, content={"message": "File uploaded successfully"})
        else:
            return JSONResponse(status_code=402, content={"message": f"Failed to download file. Status code: {response.status_code}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
