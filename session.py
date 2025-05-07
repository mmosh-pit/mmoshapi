from langchain.memory import ConversationBufferMemory
class SessionMemoryManager:
    def __init__(self):
        self.sessions = {}

    def get_memory(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )
        return self.sessions[session_id]