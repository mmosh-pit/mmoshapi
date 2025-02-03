

from google.generativeai import GenerationConfig, GenerativeModel
from google.generativeai.protos import (
    Content,
)
from typing import AsyncGenerator, List
from dataclasses import dataclass

from crud.chat import get_relevant_context
from utils.variable_constant.vertex_google import ( model_id)
from utils.variable_constant.prompt import SYSTEM_PROMPT
from .chat import save_chat_message


async def create_live_session():
    """Create a new live session with appropriate configuration."""

    
    # Generation config
    generation_config = GenerationConfig(
        temperature=0,
        top_p=0.8,
        top_k=40,
        max_output_tokens=2048,

    )

    # Create model with the config
    model = GenerativeModel(model_id, generation_config=generation_config ,) 
   
    # Start a chat session (pass the system prompt here if required by a specific method)
    chat = model.start_chat()
    
    return chat

async def generate_stream(
    prompt: str,
    username: str,
    chat_history: list[Content],
    namespaces: list[str],
    metafield: str,
    system_prompt : str
) -> AsyncGenerator[str, None]:
    """Generate streaming responses using Gemini model."""
    try:

        system_prompt = SYSTEM_PROMPT if not system_prompt else system_prompt
        # Get the chat session
        chat = await create_live_session()
        
        try:
            # Get relevant context if needed
            context = await get_relevant_context(prompt, namespaces, metafield , system_prompt)
            
            # Create the full prompt
            if context:
                full_prompt = f"{system_prompt}\n\nContext: {context}\n\nUser Question: {prompt}\n\nAnswer:"
            else:
                full_prompt = f"{system_prompt}\n\nUser Question: {prompt}\n\nAnswer:"
            
            # Generate the streaming response
            response = chat.send_message(full_prompt, stream=True)
            
            # Process the streaming response
            full_response = []
            for chunk in response:
                if chunk.text:
                    full_response.append(chunk.text)
                    yield chunk.text
            
            # Save the conversation after completion
            if full_response:
                complete_response = "".join(full_response)
                save_chat_message(username , "system" , system_prompt)
                save_chat_message(username, "user", prompt)
                save_chat_message(username, "model", complete_response)

        except Exception as processing_error:
            print(f"Error in message processing: {str(processing_error)}")
            error_msg = "I'm having trouble processing your request. Please try again."
            yield error_msg
            save_chat_message(username , "system" , system_prompt)
            save_chat_message(username, "user", prompt)
            save_chat_message(username, "model", error_msg)
            return

    except Exception as session_error:
        print(f"Session creation error: {str(session_error)}")
        error_msg = "I apologize, but I'm unable to process that request. Please try again later."
        yield error_msg
        save_chat_message(username , "system" , system_prompt)
        save_chat_message(username, "user", prompt)
        save_chat_message(username, "model", error_msg)
        return