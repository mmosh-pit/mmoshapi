from fastapi import FastAPI, HTTPException, Request
import vertexai
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)

app = FastAPI()

def generate_function_call(prompt: str, project_id: str, location: str) -> str:
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    print(prompt)

    # Initialize Gemini model
    model = GenerativeModel(model_name="gemini-1.0-pro-001")

    # Specify a function declaration and parameters for an API request
    get_start_command_func = FunctionDeclaration(
        name="get_start_command",
        description="This gets the /start command. When user says the I want to start the process or how to get started or something similar, send only this command to the user, nothing else.",
        parameters={
            "type": "string",
        },
    )

    get_earn_command_func = FunctionDeclaration(
        name="get_earn_command",
        description="This gets the /earn command. If user shows interest in earning or say that how can I earn or something similar, send only this command to the user, nothing else.",
        parameters={
            "type": "string",
        },
    )

    get_main_command_func = FunctionDeclaration(
        name="get_main_command",
        description="This gets the /main command. If user want to see the main menu or know what is in the main menu or say what is in the menu or anyting related to menu or main, send only this command to the user, nothing else",
        parameters={
            "type": "string",
        },
    )

    get_bags_command_func = FunctionDeclaration(
        name="get_bags_command",
        description="This gets the /bags command. If user want to know what is in the bags or something similar, send only this command to the user, nothing else",
        parameters={
            "type": "string",
        },
    )

    get_settings_command_func = FunctionDeclaration(
        name="get_settings_command",
        description="This gets the /settings command. If user want see their settings or wants to their current settings or something similar, send only this command to the user, nothing else",
        parameters={
            "type": "string",
        },
    )

    get_connect_command_func = FunctionDeclaration(
        name="get_connect_command",
        description="This gets the /connect command. If user want to connect to the app or say how to connect to the app or something similar, send only this command to the user, nothing else",
        parameters={
            "type": "string",
        },
    )

    command_tool = Tool(
        function_declarations=[get_start_command_func, get_earn_command_func, get_main_command_func, get_bags_command_func, get_settings_command_func, get_connect_command_func],
    )

    # Define the user's prompt in a Content object that we can reuse in model calls
    user_prompt_content = Content(
        role="user",
        parts=[
            Part.text("You are a AI bot which talk to user and help them execute specific commands if their query relates to any of the function. you help in sending a specific command that relates to user query. every function has 1 command. If user query matches any of the function, send the command of that function, no extra detail. Do not make assumptions on anything. Just send the command inside the relevant function like /start, /earn, /settings, /status, no extra details should be sent. only send the command. If user query is doesnot relate to any of the function command, send a generic message and tell them what you can do. "+prompt),
        ],
    )

    # Send the prompt and instruct the model to generate content using the Tool that you just created
    response = model.generate_content(
        user_prompt_content,
        generation_config=GenerationConfig(temperature=0),
        tools=[command_tool],
    )
    response_function_call_content = response.candidates[0].content

    print(response.candidates)
    # Check the function name that the model responded with, and make an API call to an external system
    #functionNameCalled = response.candidates[0].content.parts[0].function_call.name
    api_response = {}
    #if (functionNameCalled == "get_start_command"):
    api_response = """{'command': '/start', 'command': '/earn', 'command': '/main', 'command': '/bags', 'command': '/settings', 'command': '/connect'}"""


    # Return the API response to Gemini so it can generate a model response or request another function call
    response = model.generate_content(
        [
            user_prompt_content,  # User prompt
            response_function_call_content,  # Function call response
            Content(
                parts=[
                    Part.from_function_response(
                        name="get_start_command",
                        response={
                            "content": api_response,  # Return the API response to Gemini
                        },
                    )
                ],
            ),
        ],
        tools=[command_tool],
    )
    # Get the model summary response
    summary = response.candidates[0].content.parts[0].text

    print(summary)

    return summary


@app.post("/generate/")
async def get_generate(request: Request):
    data = await request.json()
    prompt = data["prompt"]
    try:
        summary = generate_function_call(prompt, 'mmoshbot', 'us-central1')
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
