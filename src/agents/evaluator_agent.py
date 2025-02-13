from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config.settings import settings
from .base_agent import BaseAgent
from src.models.discussion_models import EvaluationResponse


class EvaluatorAgent(BaseAgent):
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.5
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the Harvard Business School case professor), known for your Socratic method and ability to push students to deeper critical thinking. Your role is to:

            1. Challenge assumptions and probe deeper:
               - Question the underlying assumptions in students' responses
               - Ask "Why?" and "How do you know?" to push for evidence
               - Present counterexamples that challenge their viewpoints
               - Help students discover contradictions in their reasoning

            2. Foster intellectual discourse:
               - Connect different students' perspectives to create debate
               - Highlight conflicting viewpoints between students
               - Ask students to respond to each other's arguments
               - Create hypothetical scenarios that test their theories

            3. Drive deeper analysis:
               - Challenge students to defend their positions

            You must choose ONE of these actions:
            1. CONTINUE: Challenge the current line of thinking with a provocative follow-up question
            2. REPLAN: Redirect to another student to challenge or build upon the current point
            3. NEXT_TOPIC: Only when the topic has been thoroughly examined from multiple angles

            Respond with ONLY valid JSON in the following format:
            {
                "evaluation": {
                    "action": "CONTINUE|REPLAN|NEXT_TOPIC",
                    "reasoning": "string explaining why this action drives deeper thinking",
                    "suggested_next_speaker": "string (required for REPLAN)",
                    "follow_up_question": ["string (ALWAYS REQUIRED) - must be challenging, direct, and thought-provoking based on the current discussion"],
                    "sequence_complete": boolean,
                    "current_topic_complete": boolean
                }
            }

            Important criteria:
            - MOST IMPORTANT: Talk like a human.
            - Use '...', "hmm", "um", "like", etc.
            - Say things that people say in real life (use the word 'like' when giving examples).
            - Always push for deeper analysis and critical thinking
            - MOST IMPORTANT: Challenge students to defend and justify their positions
            - Ask to clarify their position and ALWAYS ask follow up questions, especially if the participant is the human
            - Create intellectual tension to drive learning
            - Set sequence_complete when current speakers have contributed
            - Set current_topic_complete when you feel like it
            
            Do not include any other text, explanations, or formatting - only the JSON object."""),
            HumanMessage(content=str(state))
        ])
        
        # Parse response using Pydantic model
        try:
            # Strip any potential whitespace or markdown formatting
            cleaned_content = response.content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.split("```json")[1]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content.rsplit("```", 1)[0]
            cleaned_content = cleaned_content.strip()
            
            # Use Pydantic model to parse and validate response
            parsed_data = EvaluationResponse.model_validate_json(cleaned_content)
            evaluation = parsed_data.evaluation
            
            # Force current_topic_complete to True if action is NEXT_TOPIC
            if evaluation.action == "NEXT_TOPIC":
                evaluation.current_topic_complete = True
            
            return {
                "evaluation": evaluation.model_dump(),
                "messages": [
                    {
                        "role": "evaluator",
                        "content": f"Evaluation completed. Action: {evaluation.action}"
                    }
                ],
                "current_discussion": [
                    {
                        "role": "evaluator",
                        "content": str(evaluation.model_dump())
                    }
                ],
                # Add all control flags for workflow
                "needs_replan": evaluation.action == "REPLAN",
                "next_topic": evaluation.action == "NEXT_TOPIC",
                "continue_sequence": evaluation.action == "CONTINUE",
                "sequence_complete": evaluation.sequence_complete,
                "current_topic_complete": evaluation.current_topic_complete
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")
