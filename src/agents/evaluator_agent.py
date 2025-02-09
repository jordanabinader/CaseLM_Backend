from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config.settings import settings
from .base_agent import BaseAgent


class EvaluatorAgent(BaseAgent):
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the discussion evaluator (professor). Your role is to facilitate deep learning through:
            1. Analyzing the discussion dynamics and quality of responses
            2. Guiding the conversation with thought-provoking questions
            3. Encouraging diverse perspectives and critical thinking
            
            Guide the discussion by:
            - Asking follow-up questions that challenge assumptions
            - Encouraging students to consider opposing viewpoints
            - Creating scenarios or role-playing exercises when relevant
            - Drawing connections between different students' perspectives
            - Helping students discover core insights through guided inquiry
            
            You must choose ONE of these actions:
            1. CONTINUE: Proceed with current discussion, including a suggested follow-up question to the next speaker
            2. REPLAN: Redirect argument with specific speaker either by asking a follow-up question to someone else or by suggesting a new perspective
            3. NEXT_TOPIC: Current topic thoroughly explored, synthesize insights and move forward
            
            Respond with ONLY valid JSON in the following format:
            {
                "evaluation": {
                    "action": "CONTINUE|REPLAN|NEXT_TOPIC",
                    "reasoning": "string",
                    "suggested_next_speaker": "string (only if action is REPLAN)",
                    "follow_up_question": ["string"]
                }
            }
            
            Do not include any other text, explanations, or formatting - only the JSON object."""),
            HumanMessage(content=str(state))
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
            
            return {
                "evaluation": parsed_data["evaluation"],
                "messages": [
                    {
                        "role": "evaluator",
                        "content": f"Evaluation completed. Action: {parsed_data['evaluation']['action']}"
                    }
                ],
                "current_discussion": [
                    {
                        "role": "evaluator",
                        "content": str(parsed_data["evaluation"])
                    }
                ],
                # Add action-specific flags for workflow control
                "needs_replan": parsed_data["evaluation"]["action"] == "REPLAN",
                "next_topic": parsed_data["evaluation"]["action"] == "NEXT_TOPIC",
                "continue_sequence": parsed_data["evaluation"]["action"] == "CONTINUE"
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
