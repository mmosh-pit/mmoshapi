
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
    )
    return model
    

@traceable(name="generate_stream" , run_type="tool")
async def generate_stream(
    prompt: str,
    username: str,
    chat_history: list[str],
    namespaces: list[str],
    metafield: str,
    system_prompt : str,
    url : str
) -> AsyncGenerator[str, None]:
    """Generate streaming responses using Gemini model."""
    try:
        
        system_prompt = system_prompt if  system_prompt else SYSTEM_PROMPT

        # Get the chat session
        chat = await create_live_session(system_prompt)
        
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

            print(messages)

            # [(),()] list of chat history  , unlimited chat history
            
            # Generate the streaming response
            response = chat.stream(messages)

            if url == "generate":
                yield response
            else: 
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