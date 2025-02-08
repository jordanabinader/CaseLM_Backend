# src/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4-turbo-preview"  
    
    class Config:
        env_file = ".env"

settings = Settings()

# src/agents/base_agent.py
from typing import Dict, Any, List
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config.settings import settings

class BaseAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(
            model=settings.OPENAI_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.7
        )
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state with specialized prompts."""
        system_prompt = self.get_system_prompt(state)
        formatted_context = self.format_input_context(state)
        
        response = await self.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=formatted_context)
        ])
        
        return self.parse_response(response.content)