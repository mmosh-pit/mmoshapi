
from utils.library_calling.google_library import google_genai_model 
from typing import AsyncGenerator
from crud.chat import get_relevant_context
from utils.variable_constant.prompt import SYSTEM_PROMPT
from .chat import save_chat_message
from langsmith import traceable
from .chat import save_chat_message , get_chat_history

@traceable(name="create_live_session" , run_type="tool")
async def create_live_session(system_prompt):
    """Create a new live session with appropriate configuration."""
    
    # Generation config
    # generation_config = GenerationConfig(
    #     temperature=0,
    #     top_p=0.8,
    #     top_k=40,
    #     max_output_tokens=2048,

    # )

    # # Create model with the config
    # model = GenerativeModel(model_id, generation_config=generation_config ,
    #                         system_instruction=system_prompt) 
    google_genai_model.system_instruction = system_prompt
   
    # Start a chat session (pass the system prompt here if required by a specific method)
    chat = google_genai_model.start_chat()
    
    return chat

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

        print(system_prompt)
        # Get the chat session
        chat = await create_live_session(system_prompt)
        
        try:
            # Get relevant context if needed
            context = await get_relevant_context(prompt, namespaces, metafield )
            
            # Create the full prompt
            
            full_prompt = f"Context: {context}\n\nUser Question: {prompt}\n\nAnswer:" if context else  f"User Question: {prompt}\n\nAnswer:"
            
            
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
                save_chat_message(username, "user", prompt)
                save_chat_message(username, "model", complete_response)

                 # Fetch the updated chat history
                updated_chat_history = await get_chat_history(username , 5)
                print("Updated Chat History: ", updated_chat_history)

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