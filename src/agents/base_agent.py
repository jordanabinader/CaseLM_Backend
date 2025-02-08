from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config.settings import settings

class BaseAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(
            temperature=0.7,
            model=settings.openai_model,
            api_key=settings.openai_api_key
        )
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Base process method to be implemented by specific agents
        
        Args:
            state: The current state dictionary containing workflow data
            
        Returns:
            Dict[str, Any]: Updated state dictionary with required keys
        """
        # Ensure the state dictionary is not modified directly
        updated_state = state.copy()
        
        # Initialize required keys if they don't exist
        if 'discussion_plan' not in updated_state:
            updated_state['discussion_plan'] = []
            
        return updated_state
