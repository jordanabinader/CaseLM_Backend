# src/agents/executor_agent.py
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.prompts.agent_prompts import EXECUTOR_PROMPT
from src.config.settings import settings
from src.agents.base_agent import BaseAgent
import json
from src.models.discussion_models import ExecutorResponse, PersonaInfo, Assignment

class ExecutorAgent(BaseAgent):
    """Agent responsible for executing the case study discussion."""
    
    def __init__(self):
        super().__init__()  # Call parent class's __init__
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key
        )
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the next part of the discussion."""
        
        assignments = state.get("assignments", [])
        if not assignments:
            raise ValueError("No assignments found in state")
        
        latest_assignment = assignments[-1]
        
        # Handle both Pydantic model and dict/SystemMessage
        if isinstance(latest_assignment, Assignment):
            professor_statement = latest_assignment.professor_statement
            assigned_persona = latest_assignment.assigned_persona
        elif hasattr(latest_assignment, 'content'):
            # It's a SystemMessage, parse its content
            try:
                content_str = latest_assignment.content.replace("'", '"')
                assignment_data = json.loads(content_str)
                professor_statement = assignment_data.get("professor_statement", "")
                assigned_persona = assignment_data.get("assigned_persona", "")
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract directly from the string
                content = latest_assignment.content
                if "professor_statement" in content and "assigned_persona" in content:
                    import ast
                    assignment_data = ast.literal_eval(content)
                    professor_statement = assignment_data.get("professor_statement", "")
                    assigned_persona = assignment_data.get("assigned_persona", "")
                else:
                    raise ValueError(f"Could not parse assignment content: {content}")
        else:
            professor_statement = latest_assignment.get("professor_statement", "")
            assigned_persona = latest_assignment.get("assigned_persona", "")
        
        if not professor_statement or not assigned_persona:
            raise ValueError("Missing professor statement or assigned persona")
            
        personas = state.get("personas", {})
        assigned_persona_data = personas.get(assigned_persona)
        
        if not assigned_persona_data:
            raise ValueError(f"Could not find data for assigned persona: {assigned_persona}")
            
        # Handle both Pydantic model and dict
        is_human = (
            assigned_persona_data.is_human 
            if isinstance(assigned_persona_data, PersonaInfo) 
            else assigned_persona_data.get("is_human", False)
        )
        print(f"assigned_persona_data: {assigned_persona_data}")
        # Get name safely
        persona_name = (
            assigned_persona_data.name 
            if isinstance(assigned_persona_data, PersonaInfo) 
            else assigned_persona_data.get("name", assigned_persona)
        )
        
        # Check if this is a human participant
        if is_human:
            return {
                "discussion": {
                    "response": {
                        "message": "Awaiting human participant response...",
                        "speaker": assigned_persona,
                        "uuid": assigned_persona_data['uuid'],
                        "references_to_others": [],
                        "questions_raised": [],
                        "key_points": []
                    }
                },
                "awaiting_user_input": True,
                "messages": [
                    {
                        "role": "system",
                        "content": f"Please provide your response as {persona_name}:\n{professor_statement}"
                    }
                ]
            }

        # Get persona data safely
        background = (
            assigned_persona_data.background 
            if isinstance(assigned_persona_data, PersonaInfo) 
            else assigned_persona_data.get("background", "")
        )
        expertise = (
            assigned_persona_data.expertise 
            if isinstance(assigned_persona_data, PersonaInfo) 
            else assigned_persona_data.get("expertise", "")
        )
        personality = (
            assigned_persona_data.personality 
            if isinstance(assigned_persona_data, PersonaInfo) 
            else assigned_persona_data.get("personality", "")
        )
        role = (
            assigned_persona_data.role 
            if isinstance(assigned_persona_data, PersonaInfo) 
            else assigned_persona_data.get("role", "")
        )

        # Add this debug print
        print(f"UUID being passed to system prompt: {assigned_persona_data.get('uuid', 'No UUID found')}")

        response = await self.llm.ainvoke([
            SystemMessage(content=self._get_system_prompt({
                "name": persona_name,
                "background": background,
                "expertise": expertise,
                "personality": personality,
                "role": role,
                "uuid": assigned_persona_data.get('uuid', '')  # Ensure we're getting the UUID
            })),
            HumanMessage(content=self._create_prompt(
                professor_statement=professor_statement,
                current_discussion=state.get("current_discussion", []),
                persona_data=assigned_persona_data
            ))
        ])
        
        try:
            print(f"response: {response}")
            parsed_data = self._clean_and_parse_response(response.content, ExecutorResponse)
            print(f"parsed_data: {parsed_data}")
            return {
                "discussion": parsed_data.model_dump(),
                "awaiting_user_input": False
            }
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")

    def _get_system_prompt(self, persona_data: Dict[str, Any]) -> str:
        return f"""You are {persona_data['name']}, with the following characteristics:
        Background: {persona_data['background']}
        Expertise: {persona_data['expertise']}
        Personality: {persona_data['personality']}
        Role: {persona_data['role']}

        IMPORTANT: You MUST use this exact UUID in your response: {persona_data['uuid']}

        Respond in character to the discussion prompt. Your response must be in this EXACT JSON format:
        {{
            "response": {{
                "message": "Your response text",
                "speaker": "{persona_data['name']}",
                "uuid": "{persona_data['uuid']}",
                "references_to_others": ["names of other participants you reference"],
                "questions_raised": ["questions you pose to others"],
                "key_points": ["main points you make"]
            }}
        }}

        Guidelines:
        - MOST IMPORTANT: Talk like a human.
        - Use '...', "hmm", "um", "like", etc.
        - Say things that people say in real life (use the word 'like' when giving examples).
        - No responses should be more than 2-3 sentences.
        - The UUID field must contain exactly: {persona_data['uuid']}
        - Stay in character
        - Reference others' points 
        - Be concise straight to the point
        - EXTREMELY casual and HUMAN 
        
        Do not include any other text, explanations, or formatting - only the JSON object."""

    def _create_prompt(self, professor_statement: str, current_discussion: list, persona_data: Dict[str, Any]) -> str:
        # Convert AIMessages to dict format
        formatted_discussion = []
        for entry in current_discussion:
            if hasattr(entry, 'content'):  # It's an AIMessage or similar
                formatted_entry = {
                    "content": entry.content,
                    "role": getattr(entry, 'role', 'unknown'),
                    "speaker": getattr(entry, 'speaker', getattr(entry, 'role', 'unknown')),
                    "references_to_others": getattr(entry, 'references_to_others', []),
                    "questions_raised": getattr(entry, 'questions_raised', []),
                    "key_points": getattr(entry, 'key_points', [])
                }
                formatted_discussion.append(formatted_entry)
            else:  # It's already a dict
                formatted_discussion.append(entry)

        return f"""Professor's Question: {professor_statement}

Current Discussion:
{json.dumps(formatted_discussion, indent=2)}

Respond as {persona_data['name']} to the professor's question, taking into account the current discussion context."""

    def _format_discussion_history(self, discussion: List[Dict[str, Any]]) -> str:
        """Format the discussion history for context."""
        formatted_history = []
        for entry in discussion:
            # Handle both dict and Message objects
            if hasattr(entry, 'content'):
                # It's a Message object
                content = entry.content
                # Try to get role/speaker from the Message object
                speaker = getattr(entry, 'speaker', getattr(entry, 'role', 'Unknown'))
            else:
                # It's a dictionary
                content = entry.get("content", "")
                speaker = entry.get("speaker", entry.get("role", "Unknown"))
            
            formatted_history.append(f"{speaker}: {content}")
        
        return "\n".join(formatted_history)

    def _format_personas(self, personas: List[Dict[str, Any]]) -> str:
        """Format the personas for the prompt."""
        formatted_personas = []
        for persona in personas:
            formatted_persona = (
                f"Name: {persona['name']}\n"
                f"Title: {persona['title']}\n"
                f"Background: {persona['background']}\n"
                f"Expertise: {', '.join(persona['expertise'])}\n"
                f"Traits: {', '.join(persona['traits'])}\n"
            )
            formatted_personas.append(formatted_persona)
        
        return "\n\n".join(formatted_personas)