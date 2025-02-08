from typing import Dict, Any, List
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

class BaseAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(temperature=0.7)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Base process method to be implemented by specific agents"""
