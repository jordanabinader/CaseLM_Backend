from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config.settings import settings
from .base_agent import BaseAgent


class ReplanAgent(BaseAgent):
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        suggested_next_speaker = state.get("suggested_next_speaker", "")
        if not suggested_next_speaker:
            raise ValueError("No suggested next speaker provided for replanning")

        # Get the follow-up question from the latest evaluation with better error handling
        evaluations = state.get("evaluations", [])
        if not evaluations:
            raise ValueError("No evaluations found in state for replanning")
        
        latest_evaluation = evaluations[-1]
        follow_up_question = ""
        
        # Handle both direct access and additional_kwargs cases
        if hasattr(latest_evaluation, 'additional_kwargs'):
            follow_up_questions = latest_evaluation.additional_kwargs.get("follow_up_question", [])
        else:
            follow_up_questions = latest_evaluation.get("follow_up_question", [])
        
        if follow_up_questions:
            follow_up_question = follow_up_questions[0]
        else:
            raise ValueError("No follow-up question found in latest evaluation")

        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the Replanner, responsible for redirecting the discussion sequence.
            Your role is to:
            1. Position the suggested speaker to directly address the previous point
            2. Create a sequence that builds on the discussion
            3. Use the provided follow-up question
            
            You must respond with ONLY valid JSON in the following format:
            {
                "updated_plan": {
                    "sequences": [
                        {
                            "topic_index": int,
                            "persona_sequence": ["persona_id1", "persona_id2", "persona_id3"],
                            "follow_up_question": "string - use the provided follow-up question"
                        }
                    ],
                    "status": "replanned"
                }
            }
            
            The persona_sequence MUST start with the suggested next speaker.
            Use the exact follow-up question provided - do not modify it.
            Do not include any other text, explanations, or formatting - only the JSON object."""),
            HumanMessage(content=f"""Replan the discussion sequence with the following context:
                Current plan: {state['discussion_plan']}
                Suggested next speaker: {suggested_next_speaker}
                Follow-up question: {follow_up_question}
                Current discussion: {state.get('current_discussion', [])}""")
        ])
        
        # Parse JSON response with error handling
        import json
        try:
            # Strip any potential whitespace or markdown formatting
            cleaned_content = response.content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.split("```json")[1]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content.rsplit("```", 1)[0]
            cleaned_content = cleaned_content.strip()
            
            parsed_data = json.loads(cleaned_content)
            
            # Verify the suggested speaker is actually next in the sequence
            first_sequence = parsed_data["updated_plan"]["sequences"][0]
            if first_sequence["persona_sequence"][0] != suggested_next_speaker:
                raise ValueError(f"Replan failed: Next speaker should be {suggested_next_speaker}, but got {first_sequence['persona_sequence'][0]}")
            
            # Force the follow-up question to be the one from the evaluator
            first_sequence["follow_up_question"] = follow_up_question
            
            return {
                "updated_plan": parsed_data["updated_plan"],
                "messages": [
                    {
                        "role": "replanner",
                        "content": f"Discussion sequence replanned. Next speaker: {suggested_next_speaker}"
                    }
                ]
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
