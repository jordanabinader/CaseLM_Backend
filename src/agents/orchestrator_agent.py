# src/agents/orchestrator_agent.py
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.prompts.agent_prompts import ORCHESTRATOR_PROMPT
from src.config.settings import settings

class OrchestratorAgent:
    """Agent responsible for orchestrating the case study discussion."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the current state and determine next steps."""
        
        # Format state for prompt
        state_summary = self._format_state(state)
        
        # Get orchestrator's decision
        response = await self.llm.ainvoke([
            SystemMessage(content=ORCHESTRATOR_PROMPT.format(state=state_summary)),
            HumanMessage(content="What should be the next step in this discussion?")
        ])
        
        # Parse the response
        next_step = self._parse_next_step(response.content)
        
        return {
            "next_step": next_step["step"],
            "messages": [
                {
                    "role": "orchestrator",
                    "content": f"Moving to step: {next_step['step']}. {next_step['reasoning']}"
                }
            ]
        }
    
    def _format_state(self, state: Dict[str, Any]) -> str:
        """Format the state for the prompt."""
        return f"""
Current Step: {state.get('current_step', 'Not started')}
Discussion Plan: {len(state.get('discussion_plan', {}))} points planned
Messages: {len(state.get('messages', []))} messages exchanged
Insights: {len(state.get('insights', []))} insights gathered
Complete: {state.get('complete', False)}
"""
    
    def _parse_next_step(self, response: str) -> Dict[str, str]:
        """Parse the LLM response to determine next step."""
        # For now, simple parsing - could be made more robust
        if "plan" in response.lower():
            return {
                "step": "plan",
                "reasoning": "Need to create or update the discussion plan."
            }
        elif "execute" in response.lower():
            return {
                "step": "execute",
                "reasoning": "Ready to execute next discussion point."
            }
        else:
            return {
                "step": "complete",
                "reasoning": "Discussion objectives have been met."
            }