from typing import Dict, Any, TypeVar, Type
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config.settings import settings
import json
import re

T = TypeVar('T', bound=BaseModel)

class BaseAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(
            temperature=0.7,
            model=settings.openai_model,
            api_key=settings.openai_api_key
        )
    
    def _clean_and_parse_response(self, response: str, model_class: Type[T]) -> T:
        """Clean LLM response and parse it with a Pydantic model.
        
        Args:
            response: Raw response string from LLM
            model_class: Pydantic model class to parse the response
            
        Returns:
            Parsed Pydantic model instance
            
        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # First, clean the string of any markdown and get just the JSON content
            cleaned_content = response.strip()
            if "```json" in cleaned_content:
                cleaned_content = cleaned_content.split("```json")[1]
            elif "```" in cleaned_content:
                cleaned_content = cleaned_content.split("```")[1]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content.rsplit("```", 1)[0]
            cleaned_content = cleaned_content.strip()
            
            # Remove any control characters
            cleaned_content = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_content)
            
            # Try to parse as JSON first to catch any JSON syntax errors
            try:
                json_dict = json.loads(cleaned_content)
            except json.JSONDecodeError as json_err:
                print(f"JSON Decode Error: {json_err}")
                print(f"Problematic content: {cleaned_content}")
                raise
                
            # Now parse with Pydantic
            return model_class.model_validate(json_dict)
            
        except Exception as e:
            print(f"Failed to parse response: {str(e)}")
            print(f"Original response: {response}")
            print(f"Cleaned content: {cleaned_content}")
            raise ValueError(f"Failed to parse LLM response: {e}")
    
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
