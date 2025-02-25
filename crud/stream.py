
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from typing import AsyncGenerator
from crud.chat import get_relevant_context
from utils.variable_constant.vertex_google import (project_id  ,location , model_id) 
import os
from utils.variable_constant.prompt import SYSTEM_PROMPT
from langsmith import traceable
from .chat import save_chat_message , get_chat_history
@traceable(name="create_live_session" , run_type="tool")
async def create_live_session(system_prompt):
    """Create a new live session with appropriate configuration."""
    
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
        system_prompt = system_prompt
    )
    return model
    

@traceable(name="generate_stream" , run_type="tool")
async def generate_stream(
    prompt: str,
    username: str,
    chat_history: list[str],
    namespaces: list[str],
    metafield: str,
    system_prompt : str
) -> AsyncGenerator[str, None]:
    """Generate streaming responses using Gemini model."""
    try:
        
 
        system_prompt = SYSTEM_PROMPT if not system_prompt else system_prompt

        # Get the chat session
        chat = await create_live_session(system_prompt)
        
        try:
            # Get relevant context if needed
            context = await get_relevant_context(prompt, namespaces, metafield )
            
            # Create the full prompt
            
            full_prompt = f"Context: {context}\n\nUser Question: {prompt}\n\nAnswer:" if context else  f"User Question: {prompt}\n\nAnswer:"
            
            
            # Generate the streaming response
            response = chat.stream(full_prompt)
            # breakpoint()re
            # Process the streaming response
            full_response = []
            for chunk in response:
                text = chunk.content  # Extract the text from the AIMessageChunk
                if text:
                    full_response.append(text)
                    yield text
            
            # Save the conversation after completion
            if full_response:
                complete_response = "".join(full_response)
                save_chat_message(username, "user", prompt)
                save_chat_message(username, "model", complete_response)

                 # Fetch the updated chat history
                updated_chat_history = await get_chat_history(username , 5)
                # print("Updated Chat History: ", updated_chat_history)
                

        except Exception as processing_error:
            print(f"Error in message processing: {str(processing_error)}")
            error_msg = "I'm having trouble processing your request. Please try again."
            yield error_msg
            save_chat_message(username, "user", prompt)
            save_chat_message(username, "model", error_msg)
            return

    except Exception as session_error:
        print(f"Session creation error: {str(session_error)}")
        error_msg = "I apologize, but I'm unable to process that request. Please try again later."
        yield error_msg
        save_chat_message(username, "user", prompt)
        save_chat_message(username, "model", error_msg)
        return