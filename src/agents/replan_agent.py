from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config.settings import settings
from .base_agent import BaseAgent
from src.models.discussion_models import ReplanResponse, PersonaInfo


class ReplanAgent(BaseAgent):
    def __init__(self):
        super().__init__()  # Call parent class's __init__
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        suggested_next_speaker = state.get("suggested_next_speaker", "")
        if not suggested_next_speaker:
            raise ValueError("No suggested next speaker provided for replanning")

        personas = state.get("personas", {})
        if not personas:
            raise ValueError("No personas found in state")

        # If suggested_next_speaker is already a participant_id, use it directly
        if suggested_next_speaker.startswith("participant_"):
            participant_id = suggested_next_speaker
        else:
            # Find the participant_id for the suggested speaker name
            participant_id = None
            for pid, persona in personas.items():
                persona_name = (
                    persona.name 
                    if isinstance(persona, PersonaInfo) 
                    else persona.get("name", "")
                )
                if persona_name == suggested_next_speaker:
                    participant_id = pid
                    break
        
            if not participant_id:
                raise ValueError(f"Could not find participant_id for speaker: {suggested_next_speaker}")

        # Get the follow-up question from the latest evaluation
        evaluations = state.get("evaluations", [])
        if not evaluations:
            raise ValueError("No evaluations found in state for replanning")
        
        latest_evaluation = evaluations[-1]
        follow_up_questions = (
            latest_evaluation.additional_kwargs.get("follow_up_question", [])
            if hasattr(latest_evaluation, 'additional_kwargs')
            else latest_evaluation.get("follow_up_question", [])
        )
        
        if not follow_up_questions:
            raise ValueError("No follow-up question found in latest evaluation")
        
        follow_up_question = follow_up_questions[0]
        
        # Get speaker name safely
        persona_data = personas[participant_id]
        speaker_name = (
            persona_data.name 
            if isinstance(persona_data, PersonaInfo) 
            else persona_data.get("name", participant_id)
        )

        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the Replanner. Your role is to:
            1. Replan the discussion sequence
            2. Ensure the specified next speaker is first
            3. Maintain logical flow of conversation
            4. Include the provided follow-up question
            
            You must respond with ONLY valid JSON in the following format:
            {
                "updated_plan": {
                    "sequences": [
                        {
                            "topic_index": 0,
                            "persona_sequence": ["participant_id1", "participant_id2"],
                            "follow_up_question": "string"
                        }
                    ],
                    "status": "replanned"
                },
                "messages": [
                    {
                        "role": "replanner",
                        "content": "Discussion sequence replanned successfully"
                    }
                ]
            }
            
            Do not include any other text, explanations, or formatting - only the JSON object."""),
            HumanMessage(content=f"""Replan the discussion sequence with:
                Current plan: {state['discussion_plan']}
                Required first speaker (participant_id): {participant_id} (Name: {speaker_name})
                Follow-up question: {follow_up_question}
                Current discussion: {state.get('current_discussion', [])}""")
        ])
        
        try:
            parsed_data = self._clean_and_parse_response(response.content, ReplanResponse)
            
            # Verify the suggested speaker is actually next in the sequence
            first_sequence = parsed_data.updated_plan.sequences[0]
            if first_sequence.persona_sequence[0] != participant_id:
                raise ValueError(f"Replan failed: Next speaker should be {participant_id}")
            
            # Force the follow-up question to be the one from the evaluator
            first_sequence.follow_up_question = follow_up_question
            
            return {
                "updated_plan": parsed_data.updated_plan.model_dump(),
                "messages": [
                    {
                        "role": "replanner",
                        "content": f"Discussion sequence replanned. Next speaker: {speaker_name}"
                    }
                ]
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")
