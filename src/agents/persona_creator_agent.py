from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from .base_agent import BaseAgent
from src.config.settings import settings
import json

class PersonaCreatorAgent(BaseAgent):
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key
        )

    def _get_system_prompt(self) -> str:
        return """You are responsible for creating personas for a case discussion. 
        Create a mix of AI and human participants, with at least one human participant.
        
        You must respond with ONLY valid JSON in the following format:
        {
            "personas": {
                "persona_id": {
                    "name": "string",
                    "background": "string",
                    "expertise": "string",
                    "personality": "string",
                    "is_human": boolean,
                    "role": "string"
                }
            }
        }
        
        Guidelines:
        - Create 3-5 personas total
        - Include at least one human participant (is_human: true)
        - persona_id should be a simple identifier like "participant_1"
        - Ensure diverse backgrounds and perspectives
        - Make backgrounds relevant to the case content
        
        Do not include any other text, explanations, or formatting - only the JSON object."""

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        case_content = state.get("case_content", "")
        if not case_content:
            raise ValueError("No case content provided")

        response = await self.llm.ainvoke([
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=f"Create personas for this case: {case_content}")
        ])

        # Parse JSON response with error handling
        try:
            cleaned_content = response.content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.split("```json")[1]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content.rsplit("```", 1)[0]
            cleaned_content = cleaned_content.strip()
            
            parsed_data = json.loads(cleaned_content)
            
            # Validate that at least one human persona exists
            has_human = any(persona.get("is_human", False) 
                          for persona in parsed_data["personas"].values())
            if not has_human:
                raise ValueError("No human personas created - at least one is required")
            
            return {
                "personas": parsed_data["personas"],
                "messages": [
                    {
                        "role": "system",
                        "content": "Personas created successfully"
                    }
                ]
            }
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")