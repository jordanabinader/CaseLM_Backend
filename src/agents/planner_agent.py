# src/agents/planner_agent.py
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config.settings import settings
from src.models.discussion_models import PlannerResponse, DiscussionPlanSequence
from src.agents.base_agent import BaseAgent

class PlannerAgent(BaseAgent):
    """Agent responsible for planning the case study discussion."""
    
    def __init__(self):
        super().__init__()  # Call parent class's __init__
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update the discussion plan."""
        # Get topics safely, handling both Pydantic model and dict
        topics = state.get('topics', {})
        if hasattr(topics, 'model_dump'):
            topics = topics.model_dump()
            
        # Get personas safely, handling both Pydantic model and dict
        personas = state.get('personas', {})
        if hasattr(personas, 'model_dump'):
            personas = personas.model_dump()

        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the Planner, responsible for determining the sequence of personas in the discussion of each topic.
            Your role is to create an engaging discussion flow by ordering the personas in a way that builds meaningful dialogue and insights.
            DO NOT include the professor in the sequence, but make sure to include all other personas.
            You must respond with ONLY valid JSON in the following format:
            {
                "plan": {
                    "sequences": [
                        {
                            "topic_index": int,
                            "persona_sequence": ["persona_id1", "persona_id2", "persona_id3"],
                            "follow_up_question": "string"
                        }
                    ],
                    "status": "created"
                }
            }
            
            The persona_sequence should list the IDs of personas in the order they should speak.
            Do not include any other text, explanations, or formatting - only the JSON object."""),
            HumanMessage(content=f"""Create a discussion sequence for each topic.
                Case content: {state['case_content']}
                Topics: {topics}
                Available personas: {personas}""")
        ])
        
        try:
            parsed_data = self._clean_and_parse_response(response.content, PlannerResponse)
            
            # Validate sequences
            for sequence in parsed_data.plan.sequences:
                if not isinstance(sequence, DiscussionPlanSequence):
                    raise ValueError(f"Invalid sequence format: {sequence}")
                if not sequence.persona_sequence:
                    raise ValueError("Empty persona sequence not allowed")
            
            return {
                "plan": parsed_data.plan.model_dump(),
                "messages": [
                    {
                        "role": "planner",
                        "content": f"Discussion sequences created for {len(parsed_data.plan.sequences)} topics."
                    }
                ]
            }
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")