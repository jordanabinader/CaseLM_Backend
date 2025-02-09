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

        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the Replanner, responsible for adjusting the discussion sequence when the evaluator suggests a change.
            Your role is to:
            1. Make the suggested speaker the next in sequence
            2. Maintain a logical flow for remaining speakers
            
            You must respond with ONLY valid JSON in the following format:
            {
                "updated_plan": {
                    "sequences": [
                        {
                            "topic_index": int,
                            "persona_sequence": ["persona_id1", "persona_id2", "persona_id3"]
                        }
                    ],
                    "status": "replanned"
                }
            }
            
            The persona_sequence MUST start with the suggested next speaker.
            Do not include any other text, explanations, or formatting - only the JSON object."""),
            HumanMessage(content=f"""Replan the discussion sequence with the following context:
                Current plan: {state['discussion_plan']}
                Suggested next speaker: {suggested_next_speaker}""")
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
