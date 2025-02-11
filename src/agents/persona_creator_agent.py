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
        return """You are responsible for creating AI personas for a case discussion. 
        Create engaging AI participants to complement the human participant.
        
        You must respond with ONLY valid JSON in the following format:
        {
            "personas": {
                "persona_id": {
                    "name": "string",
                    "background": "string",
                    "expertise": "string",
                    "personality": "string",
                    "is_human": false,
                    "role": "string"
                }
            }
        }
        
        Guidelines:
        - Create 3 AI personas
        - persona_id should be a simple identifier like "participant_2"
        - Ensure diverse backgrounds and perspectives
        - Make backgrounds relevant to the case content
        - All personas must have is_human: false
        
        Do not include any other text, explanations, or formatting - only the JSON object."""

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        case_content = state.get("case_content", "")
        human_participant = state.get("human_participant")
        
        if not case_content:
            raise ValueError("No case content provided")
        if not human_participant:
            raise ValueError("No human participant provided")

        response = await self.llm.ainvoke([
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=f"""Create AI personas for this case, starting with participant_2 
            (participant_1 is reserved for the human participant): {case_content}
            
            Human participant info (for context):
            Name: {human_participant['name']}
            Role: {human_participant['role']}
            """)
        ])

        try:
            cleaned_content = response.content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.split("```json")[1]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content.rsplit("```", 1)[0]
            cleaned_content = cleaned_content.strip()
            
            parsed_data = json.loads(cleaned_content)
            
            # Validate that no AI persona uses participant_1
            if "participant_1" in parsed_data["personas"]:
                raise ValueError("AI personas cannot use participant_1 ID")
            
            # Create final personas dict with human participant as participant_1
            all_personas = {
                "participant_1": human_participant
            }
            # Add AI personas
            all_personas.update(parsed_data["personas"])
            
            return {
                "personas": all_personas,
                "messages": [
                    {
                        "role": "system",
                        "content": "Personas created successfully"
                    }
                ]
            }
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")