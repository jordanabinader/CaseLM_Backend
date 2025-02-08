from typing import Dict, Any, List
from .base_agent import BaseAgent

class SummarizerAgent(BaseAgent):
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the discussion summarizer. Your role is to:
            1. Synthesize key points from the discussion
            2. Highlight important insights
            3. Track evolving perspectives
            4. Maintain context for next steps
            
            Summarize the current discussion state."""),
            HumanMessage(content=str(state))
        ])
        
        return {
            "summary": response.content,
            "key_insights": [],  # Key insights identified
            "next_topics": []  # Suggested next topics
        }