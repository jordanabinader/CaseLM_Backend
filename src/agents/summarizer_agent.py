from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.agents.base_agent import BaseAgent
from src.config.settings import settings
from src.models.discussion_models import SummaryResponse, DiscussionResponse

class SummarizerAgent(BaseAgent):
    def __init__(self):
        super().__init__()  # Call parent class's __init__
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Format the discussion history for summarization
        current_discussion = state.get("current_discussion", [])
        formatted_discussion = self._format_discussion_entries(current_discussion)
        
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the discussion summarizer. Your role is to:
            1. Synthesize key points from the discussion
            2. Highlight important insights
            3. Track evolving perspectives
            4. Maintain context for next steps
            
            You must respond with ONLY valid JSON in the following format:
            {
                "summary": {
                    "key_points": ["string"],
                    "insights": ["string"],
                    "evolving_perspectives": ["string"],
                    "next_steps": ["string"],
                    "overall_summary": "string"
                }
            }
            
            Do not include any other text, explanations, or formatting - only the JSON object."""),
            HumanMessage(content=f"Summarize the following discussion:\n\n{formatted_discussion}")
        ])
        
        try:
            parsed_data = self._clean_and_parse_response(response.content, SummaryResponse)
            
            return {
                "summary": parsed_data.summary.model_dump(),
                "messages": [
                    {
                        "role": "summarizer",
                        "content": f"Summary created with {len(parsed_data.summary.key_points)} key points."
                    }
                ]
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")
            
    def _format_discussion_entries(self, discussion: List[Dict[str, Any]]) -> str:
        """Format discussion entries, handling both dict and Pydantic models."""
        formatted_entries = []
        
        for entry in discussion:
            if isinstance(entry, DiscussionResponse):
                formatted_entry = {
                    "speaker": entry.speaker,
                    "message": entry.message,
                    "key_points": entry.key_points,
                    "references_to_others": entry.references_to_others,
                    "questions_raised": entry.questions_raised
                }
            elif hasattr(entry, 'content'):  # Message object
                formatted_entry = {
                    "speaker": getattr(entry, 'speaker', getattr(entry, 'role', 'unknown')),
                    "message": entry.content,
                    "key_points": getattr(entry, 'key_points', []),
                    "references_to_others": getattr(entry, 'references_to_others', []),
                    "questions_raised": getattr(entry, 'questions_raised', [])
                }
            else:  # Dictionary
                formatted_entry = {
                    "speaker": entry.get("speaker", entry.get("role", "unknown")),
                    "message": entry.get("content", entry.get("message", "")),
                    "key_points": entry.get("key_points", []),
                    "references_to_others": entry.get("references_to_others", []),
                    "questions_raised": entry.get("questions_raised", [])
                }
            
            formatted_entries.append(
                f"[{formatted_entry['speaker']}]: {formatted_entry['message']}\n"
                f"Key Points: {', '.join(formatted_entry['key_points'])}\n"
                f"Questions Raised: {', '.join(formatted_entry['questions_raised'])}\n"
                f"References: {', '.join(formatted_entry['references_to_others'])}\n"
            )
        
        return "\n".join(formatted_entries)

    def _clean_and_parse_response(self, response_content: str, model: type) -> SummaryResponse:
        cleaned_content = response_content.strip()
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content.split("```json")[1]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content.rsplit("```", 1)[0]
        cleaned_content = cleaned_content.strip()
        
        # Parse with Pydantic
        parsed_data = model.model_validate_json(cleaned_content)
        
        return parsed_data