# src/agents/executor_agent.py
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.prompts.agent_prompts import EXECUTOR_PROMPT
from src.config.settings import settings
from .base_agent import BaseAgent
import json

class ExecutorAgent(BaseAgent):
    """Agent responsible for executing the case study discussion."""
    
    def __init__(self):
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
        professor_statement = latest_assignment.get("professor_statement", "")
        assigned_persona = latest_assignment.get("assigned_persona", "")
        
        if not professor_statement or not assigned_persona:
            raise ValueError("Missing professor statement or assigned persona")
            
        personas = state.get("personas", {})
        assigned_persona_data = personas.get(assigned_persona)
        
        if not assigned_persona_data:
            raise ValueError(f"Could not find data for assigned persona: {assigned_persona}")
            
        # Check if this is a human participant
        if assigned_persona_data.get("is_human", False):
            return {
                "discussion": {
                    "response": {
                        "message": "Awaiting human participant response...",
                        "speaker": assigned_persona,
                        "references_to_others": [],
                        "questions_raised": [],
                        "key_points": []
                    }
                },
                "awaiting_user_input": True,
                "messages": [
                    {
                        "role": "system",
                        "content": f"Please provide your response as {assigned_persona_data['name']}:\n{professor_statement}"
                    }
                ]
            }

        # For AI participants, generate a response using the LLM
        response = await self.llm.ainvoke([
            SystemMessage(content=self._get_system_prompt(assigned_persona_data)),
            HumanMessage(content=self._create_prompt(
                professor_statement=professor_statement,
                current_discussion=state.get("current_discussion", []),
                persona_data=assigned_persona_data
            ))
        ])
        
        try:
            cleaned_content = response.content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.split("```json")[1]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content.rsplit("```", 1)[0]
            cleaned_content = cleaned_content.strip()
            
            parsed_data = json.loads(cleaned_content)
            
            return {
                "discussion": parsed_data,
                "awaiting_user_input": False
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")

    def _get_system_prompt(self, persona_data: Dict[str, Any]) -> str:
        return f"""You are {persona_data['name']}, with the following characteristics:
        Background: {persona_data['background']}
        Expertise: {persona_data['expertise']}
        Personality: {persona_data['personality']}
        Role: {persona_data['role']}

        Respond in character to the discussion prompt. Your response must be in JSON format:
        {{
            "response": {{
                "message": "Your response text",
                "speaker": "{persona_data['name']}",
                "references_to_others": ["names of other participants you reference"],
                "questions_raised": ["questions you pose to others"],
                "key_points": ["main points you make"]
            }}
        }}

        Guidelines:
        - Stay in character
        - Reference others' points when relevant
        - Ask thoughtful questions
        - Make clear, substantive points
        - Be concise but thorough
        
        Do not include any other text, explanations, or formatting - only the JSON object."""

    def _create_prompt(self, professor_statement: str, current_discussion: list, persona_data: Dict[str, Any]) -> str:
        return f"""Professor's Question: {professor_statement}

Current Discussion:
{json.dumps(current_discussion, indent=2)}

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