
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_xai import ChatXAI
from typing import AsyncGenerator
from crud.chat import get_relevant_context
from utils.variable_constant.vertex_google import (project_id  ,location , model_id) 
import os
from utils.variable_constant.prompt import SYSTEM_PROMPT
from langsmith import traceable
from .chat import save_chat_message , get_chat_history

model_name_groq = ["deepseek-r1-distill-llama-70b" , "deepseek-r1-distill-qwen-32b", "qwen-2.5-32b"  ,"qwen-qwq-32b" , "mistral-saba-24b",
                   "llama-3.2-1b-preview" , "llama-3.2-3b-preview","llama-3.3-70b-versatile","llama3-70b-8192","llama-3.1-8b-instant","llama3-8b-8192",
                  "mixtral-8x7b-32768","gemma2-9b-it",]

gemini_model_list = ["gemini-2.0-flash" , "gemini-2.0-flash-lite" , "gemini-2.0-pro-exp-02-05" , "gemini-1.5-flash" , "gemini-1.5-flash-8b" , "gemini-1.5-pro"]

anthropic_model_list = ["claude-3-7-sonnet-latest" , "claude-3-5-sonnet-latest" , 
                      "claude-3-5-sonnet-20240620" , "claude-3-opus-latest"]

open_ai_model_list  = [
    "gpt-4.5-preview",
    "gpt-4o",
    "chatgpt-4o-latest"
]

xai_model_list = ["grok-2-1212"]


@traceable(name="create_live_session" , run_type="tool")

async def create_live_session(model_name: str):
    """Create a new live session with appropriate configuration."""

    if model_name in gemini_model_list:
        model_id = model_name
    
        # breakpoint()
        model = ChatGoogleGenerativeAI(
            google_api_key=os.getenv('GENAI_API_KEY'),
            model=model_id,  # Specify the appropriate Gemini model (e.g., text-bison)
            temperature=0,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
            project_id=project_id,  # Set your Google Cloud Project ID
            location=location  ,# Set your Google Cloud Region
        )
    
    elif model_name in open_ai_model_list:
        model = ChatOpenAI(
            model=model_name,  # e.g., "gpt-3.5-turbo" or "gpt-4"
            temperature=0,
            max_tokens=2048,
            timeout=None,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    elif model_name in anthropic_model_list:
        model = ChatAnthropic(
            model=model_name,  # e.g., "claude-v1"
            temperature=0,
            max_tokens=2048,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    
    elif model_name in model_name_groq:
        # Hypothetical integration for Grok / X AI
        model = ChatGroq(
            api_key=os.getenv("GROK_API_KEY"),
            model=model_name,
            temperature=0,
            max_tokens=2048
        )
    elif model_name in xai_model_list:
        model = ChatXAI(
            xai_api_key=os.getenv("XAI_API_KEY"),
            model=model_name,
        )
  
    
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model
    

@traceable(name="generate_stream" , run_type="tool")
async def generate_stream(
    model_name: str,
    prompt: str,
    username: str,
    chat_history: list[str],
    namespaces: list[str],
    metafield: str,
    system_prompt : str,
    
) -> AsyncGenerator[str, None]:
    """Generate streaming responses using Gemini model."""
    try:
        
        system_prompt = system_prompt if  system_prompt else SYSTEM_PROMPT
        
        # Get the chat session
      
        chat = await create_live_session(model_name)
        
        try:
            # Get relevant context 
            context = await get_relevant_context(prompt, namespaces, metafield )
            
            # Create the full prompt
            
            full_prompt = f"Context: {context}\n\nUser Question: {prompt}\n\nAnswer:" if context else  f"User Question: {prompt}\n\nAnswer:"
  
            messages = [
                ("system", system_prompt),
                # ("chat_history", chat_history),
            ]
            
            for chats in chat_history:
                messages.extend(tuple(chats))

            messages.append(("human", full_prompt))

            
            # Generate the streaming response
            response = chat.stream(messages)

           
            # Process the streaming response
            full_response = []
            for chunk in response:
                text = chunk.content  # Extract the text from the AIMessageChunk
                if text:
                    full_response.append(text)
                    yield text
        

        except Exception as processing_error:
            print(f"Error in message processing: {str(processing_error)}")
            error_msg = "I'm having trouble processing your request. Please try again."
            yield error_msg
            return

    except Exception as session_error:
        print(f"Session creation error: {str(session_error)}")
        error_msg = "I apologize, but I'm unable to process that request. Please try again later."
        yield error_msg
        return
    
@traceable(name="generate" , run_type="tool")
async def generate(
    model_name : str,
    prompt: str,
    username: str,
    chat_history: list[tuple[str, str]],
    namespaces: list[str],
    metafield: str,
    system_prompt: str,
):
    """Generate a complete response using the Gemini model."""
    try:
        system_prompt = system_prompt if system_prompt else SYSTEM_PROMPT

        # Create the chat session
        chat = await create_live_session(model_name)
        
        try:
            # Get relevant context if available
            context = await get_relevant_context(prompt, namespaces, metafield)
            full_prompt = (
                f"Context: {context}\n\nUser Question: {prompt}\n\nAnswer:"
                if context else f"User Question: {prompt}\n\nAnswer:"
            )

            messages = [("system", system_prompt)]
            for hist in chat_history:
                messages.append(tuple(hist))
            messages.append(("human", full_prompt))
            
            # Call chat.invoke without await since it returns a non-awaitable AIMessage
            response = chat.invoke(messages)
            return str(response.content)

        except Exception as processing_error:
            print(f"Error in message processing: {str(processing_error)}")
            return "I'm having trouble processing your request. Please try again."
            
    except Exception as session_error:
        print(f"Session creation error: {str(session_error)}")
        return "I apologize, but I'm unable to process that request. Please try again later."
    
