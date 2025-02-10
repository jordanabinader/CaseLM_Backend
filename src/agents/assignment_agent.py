from typing import Dict, Any
from src.agents.base_agent import BaseAgent
import json
from langchain.schema import SystemMessage, HumanMessage

class AssignmentAgent(BaseAgent):
    """
    Agent responsible for acting as a professor and assigning questions/transitions
    for the discussion participants.
    """
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        current_step = state["current_step"]
        discussion_plan = state["discussion_plan"]
        current_discussion = state["current_discussion"]
        topics = state["topics"]
        personas = state["personas"]
        current_sequence = state.get("current_sequence")  # Get the current sequence if it exists

        # Get the follow-up question from the current sequence if available
        follow_up_question = None
        if current_sequence and "follow_up_question" in current_sequence:
            follow_up_question = current_sequence["follow_up_question"]
        
        if follow_up_question:
            # If we have a follow-up question, use it directly
            assigned_persona = current_sequence["persona_sequence"][0]
            
            # Check if the assigned persona is a human participant
            is_human = personas.get(assigned_persona, {}).get("is_human", False)
            
            return {
                "assignment": {
                    "professor_statement": follow_up_question,
                    "assigned_persona": assigned_persona
                },
                "messages": [
                    {
                        "role": "professor",
                        "content": follow_up_question
                    }
                ],
                "awaiting_user_input": is_human  # Set flag if human persona is assigned
            }

        # Otherwise, fall back to generating a new question
        response = await self.llm.ainvoke([
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=self._create_prompt(
                current_step=current_step,
                discussion_plan=discussion_plan,
                current_discussion=current_discussion,
                topics=topics,
                personas=personas
            ))
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
                "assignment": parsed_data["assignment"],
                "messages": [
                    {
                        "role": "professor",
                        "content": parsed_data["assignment"]["professor_statement"]
                    }
                ]
            }
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")

    def _get_system_prompt(self) -> str:
        return """You are a professor leading a case discussion. Your role is to guide the discussion by asking questions and managing transitions between topics.
        You must respond with ONLY valid JSON in the following format:
        {
            "assignment": {
                "professor_statement": "The actual question or transition statement",
                "assigned_persona": "persona_id"
            }
        }
        Do not include any other text, explanations, or formatting - only the JSON object."""

    def _create_prompt(self, current_step: str, discussion_plan: Dict[str, Any],
                      current_discussion: list, topics: Dict[str, Any],
                      personas: Dict[str, Any]) -> str:
        return f"""Based on the current discussion state, determine the next logical question or transition needed.

Current Step: {current_step}
Discussion Plan: {discussion_plan}
Current Discussion History: {current_discussion}
Available Topics: {topics}
Available Personas: {personas}"""