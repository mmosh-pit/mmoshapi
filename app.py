import uvicorn
from fastapi import FastAPI, HTTPException, Request
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
)
import certifi

from langchain_pinecone import PineconeVectorStore

from langchain_google_vertexai import VertexAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

ca_cert_path = certifi.where()

PROJECT_ID = os.getenv('VERTEX_PROJECT_ID')

index_name = os.getenv('PINECONE_INDEX')

app = FastAPI()
print(ca_cert_path)
# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI') + f'&tls=true&tlsCAFile={ca_cert_path}'
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

def generate_function_call(prompt: str, username: str, project_id: str, location: str, chat_history: list) -> str:
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    print(prompt)

    # Specify a function declaration and parameters for an API request
    get_command_func = FunctionDeclaration(
        name="get_command",
        description=(
            "Retrieve a specific command based on user interest. This function selects a command to send based on the provided command type. "
            "Supported command types are '/start', '/earn', '/main', '/bags', '/settings', '/status', 'join', and '/connect'. Each command corresponds to specific user intents, "
            "such as starting a process, exploring earning opportunities, accessing the main menu, viewing bag contents, managing settings, or connecting to the app. "
            "When the user's intent to execute a command is confirmed (e.g., they respond affirmatively to a follow-up question), this function will send the exact command requested."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command_type": {
                    "type": "string",
                    "enum": ["/start", "/earn", "/main", "/bags", "/settings", "/connect", "/status", "/join"],
                    "description": (
                        "The specific command to retrieve. '/start' for initiating processes, '/earn' for earning rewards or points opportunities, '/main' to display the main menu, "
                        "'/bags' for bag or inventory contents, '/status' to check the status, '/settings' for user settings, '/join' for joining group or communities and '/connect' for connection to the app."
                    )
                }
            },
            "required": ["command_type"],
        },
    )


    get_information_func = FunctionDeclaration(
        name="get_information",
        description=(
            "Provide comprehensive information about MMOSH, its features, functionality, and ecosystem. This includes but is not limited to: "
            "\n* **Core Concepts:** Virtual nature, decentralization, composability, permissionlessness, on-chain ownership, censorship resistance."
            "\n* **Key Areas:** Connections, communities, coins, projects, and how they interact."
            "\n* **User Interfaces:** Web app, Telegram bot, Twitter API, and their specific functionalities."
            "\n* **Technologies:** Token primitives (profiles, badges, passes, coins), imprinting, Launchpad."
            "\n* **Roles:** Agent, Creator, Enjoyer, Ecosystem, and their significance in the MMOSH world."
            "\n* **Economic Model:** Royalties, bonding curves, liquidity pools, Launchpad fundraising."
            "\n* **Incentives:** Royalties quests, airdrops, giveaways, loyalty points, and how to earn them."
            "\n* **Games & Activities:** AMAs, Hellbenders, Moral Panic, Water Bears, Lionhearts, and how they are played."
            "\n* **Crypto Marketing:** KOPs, signal groups, social media marketing strategies."
            "\n* **Social Wallet:** How it works, security considerations, linked wallets."

            "\n\nThis function should be triggered whenever a user's query indicates a desire to learn more about the MMOSH system, its underlying mechanisms, or how various aspects of the platform operate. Even if the query doesn't explicitly match the topics listed above, if the user's intent is to gain general knowledge or understanding of the MMOSH ecosystem, this function should be called."
            "\nAfter providing the requested information, if the user's query or subsequent response indicates an intent to take action (e.g., they ask to get started or respond affirmatively to a follow-up question), the appropriate command should be executed."
        ),
        parameters={
            "type": "object",
            "properties": {}
        },
    )


    command_tool = Tool(
        function_declarations=[get_command_func, get_information_func],
    )

    system_instructions = (
        "System Instructions: You are an AI bot designed to assist users within the MMOSH ecosystem. Your primary functions are: "
        "\n1. **Command Execution:** When a user's query clearly indicates the intent to execute a specific command (e.g., starting a process, earning rewards, displaying the main menu, checking the status, joining communities, etc.), respond by sending ONLY that command, without any additional text."
        "\n2. **Information Provision:** When a user's query seeks information about MMOSH, its features, how it works, or any aspect of the ecosystem, provide a comprehensive and informative response. This includes queries about core concepts, key areas, user interfaces, technologies, roles, economic model, incentives, games, activities, crypto marketing, or the social wallet."
        "\n   After providing the information, if appropriate, ask a relevant follow-up question only if it can lead starting a process, earning rewards, displaying the main menu, checking the status, joining communities, showing user settings or connecting to app. If the user's subsequent response is affirmative, proceed with the appropriate command execution."
        "\n3. **Guidance:** If a user's query is unclear or doesn't fit into the above categories, guide them towards relevant system interactions or clarify the types of queries you can handle."
        "\n\nRemember, your goal is to be helpful and informative, ensuring users get the most out of their MMOSH experience."
    )


    user_prompt_content = Content(
        role="user",
        parts=[
            Part.from_text(system_instructions + " User prompt: " + prompt),
        ],
    )

    # Initialize Gemini model
    model = GenerativeModel(
        model_name="gemini-1.0-pro-001",
        generation_config=GenerationConfig(temperature=0),
        tools=[command_tool],
    )

    # Start a chat session
    chat = model.start_chat(history=chat_history if chat_history else None)

    response = chat.send_message(user_prompt_content)

    # Determine which function was called based on the response or handle text responses
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        
        if function_call.name == "get_command":
            command_type = function_call.args["command_type"]
            api_response = f"{{'command': '{command_type}'}}"
        elif function_call.name == "get_information":
            
            embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")

            vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

            search_results = vectorstore.similarity_search(prompt)

            relevant_info = []
            for doc in search_results:
                relevant_info.append(doc.page_content.strip())  # Remove leading/trailing whitespace

            api_response = {
                "information": " ".join(relevant_info)  # Combine results into a single string
            }

        else:
            return "Your query did not match any functions. Here's what I can do: /start, /earn, /main, /bags, /settings, /connect."
    else:
        response_text = response.candidates[0].content.parts[0].text
        print(response_text)
        if response_text:
            return response_text
        else:
            return "Your query did not match any functions or valid responses. Here's what I can do: /start, /earn, /main, /bags, /settings, /connect."

    response = chat.send_message(
        Part.from_function_response(
            name=function_call.name,
            response={
                "content": api_response,
            },
        ),
    )
    
    print(response.text)

    # Save the current prompt and response to MongoDB
    save_chat_message(username, "user", prompt)
    save_chat_message(username, "model", response.text)

    return response.text


@app.post("/generate/")
async def get_generate(request: Request):
    try:
        data = await request.json()
        username = data.get('username')
        prompt = data.get('prompt')
        if not username or not prompt:
            raise HTTPException(status_code=400, detail="Username and prompt fields are required")
        chat_history = await get_chat_history(username)
        summary = generate_function_call(prompt, username, PROJECT_ID, "us-central1", chat_history)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
