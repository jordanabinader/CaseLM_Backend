from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.agents.base_agent import BaseAgent
from src.config.settings import settings
from src.models.discussion_models import PersonaResponse, PersonaInfo
import json
import uuid

class PersonaCreatorAgent(BaseAgent):
    def __init__(self):
        super().__init__()
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
                "participant_id": {
                    "name": "string",
                    "background": "string",
                    "expertise": "string",
                    "personality": "string",
                    "is_human": false,
                    "role": "string",
                    "voice": "string"
                }
            },
            "professor": {
                "name": "string",
                "background": "string",
                "expertise": "string",
                "personality": "string",
                "is_human": false,
                "role": "Professor",
                "voice": "string",
                "introduction_statement": "string (one or two sentences introducing the discussion)"
            }
        }
        
        Guidelines:
        - Create exactly 3 AI personas
        - Use participant_2, participant_3, and participant_4 as participant_id
        - Ensure diverse backgrounds and perspectives
        - Make backgrounds relevant to the case content
        - All personas must have is_human: false
        - For the voice and name field, choose ONLY from these exact options:
            - "aura-asteria-en" (Asteria)
            - "aura-luna-en" (Luna)
            - "aura-stella-en" (Stella)
            - "aura-athena-en" (Athena)
            - "aura-hera-en" (Hera)
            - "aura-orion-en" (Orion)
            - "aura-arcas-en" (Arcas)
            - "aura-perseus-en" (Perseus)
            - "aura-angus-en" (Angus)
            - "aura-orpheus-en" (Orpheus)
            - "aura-helios-en" (Helios)
            - "aura-zeus-en" (Zeus)
        
        Do not include any other text, explanations, or formatting - only the JSON object."""

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        case_content = state.get("case_content", "")
        human_participant = state.get("human_participant")
        human_participant["voice"] = ""
        if not case_content:
            raise ValueError("No case content provided")
        if not human_participant:
            raise ValueError("No human participant provided")

        # Convert human participant to PersonaInfo if it's a dict
        human_persona = (
            human_participant 
            if isinstance(human_participant, PersonaInfo)
            else PersonaInfo(**human_participant)
        )

        # Create UUIDs first
        human_uuid = str(uuid.uuid4())
        ai_uuids = [str(uuid.uuid4()) for _ in range(3)]
        professor_uuid = str(uuid.uuid4())  # Add UUID for professor

        response = await self.llm.ainvoke([
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=f"""Create AI personas for this case, starting with participant_2 
            (participant_1 is reserved for the human participant): {case_content}
            
            Human participant info (for context):
            Name: {human_persona.name}
            Role: {human_persona.role}
            """)
        ])

        try:
            parsed_data = self._clean_and_parse_response(response.content, PersonaResponse)
            
            # Add UUID to professor info
            professor_info = parsed_data.professor
            professor_dict = professor_info.model_dump()
            professor_dict["uuid"] = professor_uuid
            
            # Validate that no AI persona uses participant_1
            if "participant_1" in parsed_data.personas:
                raise ValueError("AI personas cannot use participant_1 ID")
            
            # Create final personas dict with human participant
            human_persona_dict = human_persona.model_dump()
            human_persona_dict["uuid"] = human_uuid
            all_personas = {
                human_uuid: human_persona_dict
            }
            
            # Add AI personas with pre-generated UUIDs
            ai_personas = list(parsed_data.personas.values())
            for persona, uuid_str in zip(ai_personas, ai_uuids):
                persona_dict = persona.model_dump()
                persona_dict["uuid"] = uuid_str
                all_personas[uuid_str] = persona_dict
            
            return {
                "personas": all_personas,
                "professor": professor_dict,
                "messages": [
                    {
                        "role": "system",
                        "content": "Personas created successfully"
                    }
                ]
            }
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")