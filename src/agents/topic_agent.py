# src/agents/planner_agent.py
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config.settings import settings
from src.models.discussion_models import TopicResponse
from src.agents.base_agent import BaseAgent

class TopicAgent(BaseAgent):
    """Agent responsible for topic selection for the case study discussion."""
    
    def __init__(self):
        super().__init__()  # Call parent class's __init__
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update the discussion plan."""
        case_content = state.get("case_content", "")
        if not case_content:
            raise ValueError("No case content provided")

        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the Topic Planner, responsible for identifying the 3 most critical topics or core insights for this Harvard Business School case study discussion.
            Your role is to create a focused roadmap covering the 3 most important aspects that will lead to meaningful learning outcomes.
            For each of the 3 topics, you will:
            1. Identify a key aspect of the case that warrants deep discussion
            2. Develop questions that will drive meaningful debate
            3. Outline expected insights students should discover
            4. Structure the flow to build toward actionable conclusions
                          
            You must respond with ONLY valid JSON in the following format:
            {
                "plan": {
                    "topics": [
                        {
                            "title": "string",
                            "expected_insight": "string"
                        }
                    ],
                    "sequence": [0, 1, 2],
                    "status": "created"
                }
            }
            
            Ensure you provide exactly 3 topics in the response.
            Do not include any other text, explanations, or formatting - only the JSON object."""),
            HumanMessage(content=f"Develop a focused three-topic discussion plan. Include cold-call opportunities, debate questions, and key moments for insight evaluation. Create a discussion plan for this case: {case_content}")
        ])
        
        try:
            parsed_data = self._clean_and_parse_response(response.content, TopicResponse)
            
            # Validate we have exactly 3 topics
            if len(parsed_data.plan.topics) != 3:
                raise ValueError(f"Expected exactly 3 topics, got {len(parsed_data.plan.topics)}")
            
            # Validate sequence contains indices 0, 1, 2
            expected_sequence = set([0, 1, 2])
            if set(parsed_data.plan.sequence) != expected_sequence:
                raise ValueError("Topic sequence must contain exactly indices 0, 1, 2")
            
            return {
                "plan": parsed_data.plan.model_dump(),
                "messages": [
                    {
                        "role": "planner",
                        "content": f"Discussion plan created with {len(parsed_data.plan.topics)} topics."
                    }
                ]
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")