# src/agents/orchestrator_agent.py
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.prompts.agent_prompts import ORCHESTRATOR_PROMPT
from src.config.settings import settings
from src.models.discussion_models import OrchestratorResponse, DiscussionState
from src.agents.base_agent import BaseAgent

class OrchestratorAgent(BaseAgent):
    """Agent responsible for orchestrating the case study discussion."""
    
    def __init__(self):
        super().__init__()  # Call parent class's __init__
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and determine next steps."""
        
        # Format state for prompt
        state_summary = self._format_state(state)
        
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the orchestrator of a case study discussion.
            Your role is to determine the next logical step in the discussion process.
            
            You must respond with ONLY valid JSON in the following format:
            {
                "next_step": {
                    "step": "string - one of: plan, execute, complete",
                    "reasoning": "string explaining the decision"
                }
            }
            
            Do not include any other text, explanations, or formatting - only the JSON object."""),
            HumanMessage(content=f"Based on the current state, what should be the next step?\n\n{state_summary}")
        ])
        
        try:
            parsed_data = self._clean_and_parse_response(response.content, OrchestratorResponse)
            
            return {
                "next_step": parsed_data.next_step.step,
                "messages": [
                    {
                        "role": "orchestrator",
                        "content": f"Moving to step: {parsed_data.next_step.step}. {parsed_data.next_step.reasoning}"
                    }
                ]
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")
    
    def _format_state(self, state: Dict[str, Any]) -> str:
        """Format the state for the prompt, handling both dict and Pydantic models."""
        discussion_plan = state.get('discussion_plan', {})
        plan_length = (
            len(discussion_plan.sequences) 
            if hasattr(discussion_plan, 'sequences') 
            else len(discussion_plan)
        )
        
        messages = state.get('messages', [])
        messages_length = (
            len(messages) 
            if isinstance(messages, list) 
            else len(messages.dict() if hasattr(messages, 'dict') else [])
        )
        
        insights = state.get('insights', [])
        insights_length = (
            len(insights) 
            if isinstance(insights, list) 
            else len(insights.dict() if hasattr(insights, 'dict') else [])
        )
        
        return f"""
Current Step: {state.get('current_step', 'Not started')}
Discussion Plan: {plan_length} points planned
Messages: {messages_length} messages exchanged
Insights: {insights_length} insights gathered
Complete: {state.get('complete', False)}
"""