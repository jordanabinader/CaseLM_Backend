# src/agents/executor_agent.py
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.prompts.agent_prompts import EXECUTOR_PROMPT
from src.config.settings import settings

class ExecutorAgent:
    """Agent responsible for executing the case study discussion."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key
        )
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the current discussion point."""
        
        # Get current discussion point from plan
        current_point = self._get_current_point(state)
        personas = state.get("personas", {})
        
        # Get executor's output
        response = await self.llm.ainvoke([
            SystemMessage(content=EXECUTOR_PROMPT.format(
                current_point=current_point,
                personas=self._format_personas(personas)
            )),
            HumanMessage(content="Generate the discussion for this point.")
        ])
        
        # Parse insights and discussion
        discussion, insights = self._parse_response(response.content)
        
        return {
            "messages": [
                {
                    "role": "executor",
                    "content": discussion
                }
            ],
            "insights": insights
        }
    
    def _get_current_point(self, state: Dict[str, Any]) -> str:
        """Get the current discussion point from the plan."""
        plan = state.get("discussion_plan", {})
        topics = plan.get("topics", [])
        sequence = plan.get("sequence", [])
        
        if not topics or not sequence:
            return "Initial discussion point"
        
        current_idx = len(state.get("insights", []))
        if current_idx >= len(sequence):
            return "Final discussion point"
            
        return topics[sequence[current_idx]]
    
    def _format_personas(self, personas: Dict[str, Any]) -> str:
        """Format personas for the prompt."""
        if not personas:
            return "No specific personas assigned"
            
        return "\n".join([
            f"- {name}: {details.get('role', 'Unknown role')}"
            for name, details in personas.items()
        ])
    
    def _parse_response(self, response: str) -> tuple[str, List[str]]:
        """Parse the discussion and insights from the response."""
        # Simple parsing - could be made more sophisticated
        insights = []
        discussion_lines = []
        
        current_section = "discussion"
        for line in response.split('\n'):
            if "KEY INSIGHT:" in line.upper():
                current_section = "insights"
                insight = line.split(":", 1)[1].strip()
                if insight:
                    insights.append(insight)
            elif current_section == "discussion":
                discussion_lines.append(line)
            elif current_section == "insights" and line.strip():
                insights.append(line.strip())
        
        return (
            "\n".join(discussion_lines).strip(),
            insights
        )