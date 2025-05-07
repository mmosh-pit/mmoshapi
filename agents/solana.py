import os
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langsmith import traceable
from typing import AsyncGenerator
from tools.solana import solana_transfer
from session import SessionMemoryManager
from pricing.usage import PrintCost
from langsmith import traceable

session_manager = SessionMemoryManager()

@traceable(name="create_live_session" , run_type="tool")
async def create_live_session(username: str):
    return ChatOpenAI(
                model="gpt-4o",  # e.g., "gpt-3.5-turbo" or "gpt-4"
                temperature=0,
                max_tokens=2048,
                timeout=None,
                api_key=os.getenv("OPENAI_API_KEY"),
                callbacks=[PrintCost(username=username)] 
            )

@traceable(name="solana_agent" , run_type="tool")
async def solana_agent(
    prompt: str,
    username: str
) -> AsyncGenerator[str, None]:
    try:
        llm = await create_live_session(username)

        # Get or create memory for this session
        memory = session_manager.get_memory(username)

        # Dynamically build an agent per request
        agent = initialize_agent(
            tools=[solana_transfer],
            llm=llm,
            memory=memory,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

        response = await agent.arun(prompt)

        
        # Process the streaming response
        full_response = []
        # for chunk in response:
        text = response.content  # Extract the text from the AIMessageChunk
        if text:
            full_response.append(text)
            yield text

    except Exception as session_error:
        print(f"Session creation error: {str(session_error)}")
        error_msg = "I apologize, but I'm unable to process that request. Please try again later."
        yield error_msg
        return