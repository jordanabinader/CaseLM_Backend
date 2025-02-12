from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal

class Assignment(BaseModel):
    professor_statement: str
    assigned_persona: str

class DiscussionResponse(BaseModel):
    message: str
    speaker: str
    references_to_others: List[str] = []
    questions_raised: List[str] = []
    key_points: List[str] = []

class ExecutorResponse(BaseModel):
    discussion: Dict[str, DiscussionResponse]
    awaiting_user_input: bool = False
    messages: Optional[List[Dict[str, str]]] = None

class AssignmentResponse(BaseModel):
    assignment: Assignment
    assignments: List[Assignment] = []
    awaiting_user_input: bool = False
    messages: List[Dict[str, str]] = []

class EvaluationAction(BaseModel):
    action: Literal["CONTINUE", "REPLAN", "NEXT_TOPIC"]
    reasoning: str
    suggested_next_speaker: Optional[str] = None
    follow_up_question: List[str]
    sequence_complete: bool
    current_topic_complete: bool

class EvaluationResponse(BaseModel):
    evaluation: EvaluationAction

class Persona(BaseModel):
    name: str
    background: str
    expertise: str
    personality: str
    role: str
    is_human: bool = False
    voice: str

class DiscussionSequence(BaseModel):
    follow_up_question: str
    persona_sequence: List[str]

class DiscussionPlan(BaseModel):
    sequences: List[DiscussionSequence]
    topics: Dict[str, Any]

class DiscussionState(BaseModel):
    current_step: str
    discussion_plan: DiscussionPlan
    current_discussion: List[Dict[str, Any]]
    topics: Dict[str, Any]
    personas: Dict[str, Persona]
    current_sequence: Optional[DiscussionSequence] = None
    assignments: List[Assignment] = []
    messages: List[Dict[str, Any]] = []
    evaluations: List[Dict[str, Any]] = []
    awaiting_user_input: bool = False

class TopicPlan(BaseModel):
    topics: List[Dict[str, Any]]
    sequence: List[int]
    status: str

class TopicResponse(BaseModel):
    plan: TopicPlan

class DiscussionPlanSequence(BaseModel):
    topic_index: int
    persona_sequence: List[str]
    follow_up_question: Optional[str] = None

class DiscussionPlan(BaseModel):
    sequences: List[DiscussionPlanSequence]
    status: str

class PlannerResponse(BaseModel):
    plan: DiscussionPlan

class SummaryContent(BaseModel):
    key_points: List[str]
    insights: List[str]
    evolving_perspectives: List[str]
    next_steps: List[str]
    overall_summary: str

class SummaryResponse(BaseModel):
    summary: SummaryContent

class ExecutorDiscussionResponse(BaseModel):
    message: str
    speaker: str
    uuid: str
    references_to_others: List[str] = []
    questions_raised: List[str] = []
    key_points: List[str] = []

class ExecutorResponse(BaseModel):
    response: ExecutorDiscussionResponse

class OrchestratorStep(BaseModel):
    step: str
    reasoning: str

class OrchestratorResponse(BaseModel):
    next_step: OrchestratorStep

class PersonaInfo(BaseModel):
    name: str
    background: str
    expertise: str
    personality: str
    is_human: bool = False
    role: str
    voice: str

class ProfessorInfo(BaseModel):
    name: str
    background: str
    expertise: str
    personality: str
    introduction_statement: str
    voice: str
    is_human: bool = False
    role: str = "Professor"

class PersonaResponse(BaseModel):
    personas: Dict[str, PersonaInfo]
    professor: ProfessorInfo

class ReplanResponse(BaseModel):
    updated_plan: DiscussionPlan
    messages: List[Dict[str, str]] 