from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from langchain.schema import SystemMessage, HumanMessage
from src.models.discussion_models import Assignment, AssignmentResponse, PersonaInfo

class AssignmentAgent(BaseAgent):
    """
    Agent responsible for acting as a professor and assigning questions/transitions
    for the discussion participants.
    """
    
    def __init__(self):
        super().__init__()  # Call parent class's __init__

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        current_step = state["current_step"]
        discussion_plan = state["discussion_plan"]
        current_discussion = state["current_discussion"]
        topics = state["topics"]
        personas = state["personas"]
        current_sequence = state.get("current_sequence")  # Get the current sequence if it exists

        # Get the follow-up question from the current sequence if available
        if current_sequence and "follow_up_question" in current_sequence:
            follow_up_question = current_sequence["follow_up_question"]
            assigned_persona = current_sequence["persona_sequence"][0]
            
            # Check if the assigned persona is a human participant
            persona_info = personas[assigned_persona]
            is_human = (
                persona_info.is_human 
                if isinstance(persona_info, PersonaInfo) 
                else persona_info.get("is_human", False)
            )
            
            if is_human:
                assignment = Assignment(
                    professor_statement=follow_up_question,
                    assigned_persona=assigned_persona
                )
                
                # Get name safely whether it's a Pydantic model or dict
                persona_name = (
                    persona_info.name 
                    if isinstance(persona_info, PersonaInfo) 
                    else persona_info.get("name", assigned_persona)
                )
                
                response = AssignmentResponse(
                    assignment=assignment,
                    assignments=[assignment],
                    awaiting_user_input=True,
                    messages=[{
                        "role": "professor",
                        "content": f"{follow_up_question}\n\nPlease provide your response as {persona_name}."
                    }]
                )
                
                return response.model_dump()
        
        # Generate a new assignment if no follow-up question exists
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

        try:
            cleaned_content = self._clean_and_parse_response(response.content, AssignmentResponse)
            return cleaned_content.model_dump()
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")

    def _get_system_prompt(self) -> str:
        return """You are a professor leading a case discussion. Your role is to:
        1. Ask thought-provoking questions
        2. Guide the discussion flow
        3. Engage specific participants
        4. Build on previous responses

        You must respond with ONLY valid JSON in the following format:
        {
            "assignment": {
                "professor_statement": "The actual question or transition statement",
                "assigned_persona": "persona_id"
            },
            "assignments": [
                {
                    "professor_statement": "The actual question or transition statement",
                    "assigned_persona": "persona_id"
                }
            ],
            "awaiting_user_input": false,
            "messages": [
                {
                    "role": "professor",
                    "content": "Assignment created successfully"
                }
            ]
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