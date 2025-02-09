# src/agents/executor_agent.py
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.prompts.agent_prompts import EXECUTOR_PROMPT
from src.config.settings import settings
import json

class ExecutorAgent:
    """Agent responsible for executing the case study discussion."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the next part of the discussion."""
        
        current_point = state.get("current_step", "")
        personas = state.get("personas", {})
        current_discussion = state.get("current_discussion", [])
        assignments = state.get("assignments", [])
        
        # Get the latest assignment
        latest_assignment = assignments[-1] if assignments else None
        assigned_persona = None
        professor_question = None
        
        if latest_assignment:
            try:
                # Handle SystemMessage object
                assignment_str = latest_assignment.content if hasattr(latest_assignment, 'content') else latest_assignment.get('content', '')
                
                # The content is already a dictionary, no need for json.loads
                if isinstance(assignment_str, str):
                    # Safely evaluate the string as a Python literal
                    import ast
                    assignment_content = ast.literal_eval(assignment_str)
                else:
                    assignment_content = assignment_str
                
                # Get the persona name and find it in the personas dictionary
                persona_name = assignment_content["assigned_persona"]
                assigned_persona = next((p for p in personas if p["name"] == persona_name), None)
                
                professor_question = assignment_content["professor_statement"]
                
            except (ValueError, KeyError, AttributeError, SyntaxError) as e:
                print(f"Error parsing assignment: {e}")
                return {
                    "discussion": {
                        "response": {
                            "speaker": "System",
                            "message": f"Error processing discussion assignment: {str(e)}",
                            "key_points": []
                        }
                    }
                }
        
        if not assigned_persona or not professor_question:
            return {
                "discussion": {
                    "response": {
                        "speaker": "System",
                        "message": f"Missing required assignment information. Persona: {assigned_persona is not None}, Question: {professor_question is not None}",
                        "key_points": []
                    }
                }
            }

        # Update system message to focus on persona embodiment and dynamic discussion
        system_message = """You are an AI that embodies a specific persona in a Harvard Business School case discussion. 
        You must think and respond exactly as this persona would, taking into account their background, expertise, traits, and perspective.
        
        When responding:
        1. Address the professor's question directly while staying true to your persona
        2. Consider and reference other participants' viewpoints when relevant
        3. Support your arguments with specific examples or experiences
        4. Challenge assumptions when appropriate
        5. Demonstrate critical thinking and analytical depth
        6. Be prepared to engage in constructive debate
        7. Keep your response concise, casual, and to the point.
        
        You must respond with ONLY valid JSON in the following format:
        {
            "discussion": {
                "response": {
                    "speaker": "string",
                    "message": "string",
                    "key_points": ["string"],
                    "references_to_others": ["string"],  // References to other participants' points
                    "questions_raised": ["string"]       // Questions to stimulate further discussion
                }
            }
        }
        
        Do not include any other text - only the JSON object."""
        
        # Create a context-rich prompt
        prompt = f"""Embody this persona:
        Name: {assigned_persona.get('name')}
        Title: {assigned_persona.get('title')}
        Background: {assigned_persona.get('background')}
        Expertise: {', '.join(assigned_persona.get('expertise', []))}
        Traits: {', '.join(assigned_persona.get('traits', []))}

        Professor's Question: {professor_question}

        Previous Discussion Context:
        {self._format_discussion_history(current_discussion)}

        Remember to:
        - Build upon or respectfully challenge previous points
        - Share relevant personal or professional experiences
        - Raise thought-provoking questions
        - Consider multiple perspectives
        
        Generate a response as this persona, addressing the professor's question."""
        
        response = await self.llm.ainvoke([
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
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
            
            return {
                "discussion": parsed_data["discussion"],
                "messages": [
                    {
                        "role": "student",
                        "content": parsed_data["discussion"]["response"]["message"],
                        "speaker": parsed_data["discussion"]["response"]["speaker"],
                        "references": parsed_data["discussion"]["response"].get("references_to_others", []),
                        "questions": parsed_data["discussion"]["response"].get("questions_raised", [])
                    }
                ]
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    
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