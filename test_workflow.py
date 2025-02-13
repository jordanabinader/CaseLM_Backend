import asyncio
from typing import Dict, Any
from src.workflow.case_discussion_workflow import CaseDiscussionWorkflow

# Test case content
TEST_CASE = """
Case Study: AI Product Launch Strategy

Background:
TechCo is preparing to launch a new AI-powered product management tool. 
The tool uses machine learning to help product managers track feature requests, 
prioritize development, and predict user adoption.

Current Situation:
- Product is in final beta testing
- $1M in development costs invested
- 500 beta users with 80% satisfaction rate
- Competitors are 6-12 months behind
- Team of 10 engineers and 3 product managers

Key Decisions Needed:
1. Pricing strategy
2. Go-to-market approach
3. Initial target market
4. Launch timing

Questions for Discussion:
1. How should TechCo position and price this product?
2. Which market segment should they target first?
3. What risks should they consider?
4. How can they maximize early adoption?
"""

# Define the human participant
HUMAN_PARTICIPANT = {
    "name": "Jordan",  
    "background": "Research Scientist at DeepMind with experience in AI/ML products",
    "expertise": "Research in AI/ML products",
    "personality": "Technical and Strategic thinker with customer focus",
    "is_human": True,
    "role": "Research Scientist"
}

async def run_discussion():
    # Initialize the workflow
    discussion = CaseDiscussionWorkflow()
    
    # Start the discussion
    state = await discussion.run(
        case_content=TEST_CASE,
        human_participant=HUMAN_PARTICIPANT
    )
    
    # Continue the discussion until complete
    while True:
        if state.get("complete"):
            print("Discussion complete!")
            break
            
        if state.get("awaiting_user_input"):
            # Get user input
            user_response = input("Your response: ")
            # Continue the discussion with the user's response
            state = await discussion.provide_user_response(user_response)
        else:
            break

if __name__ == "__main__":
    asyncio.run(run_discussion())