from pydantic import BaseModel, EmailStr, constr, validator, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

# Enum for case status
class CaseStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"

# Base schemas
class UserBase(BaseModel):
    email: EmailStr

class CaseBase(BaseModel):
    title: constr(min_length=5, max_length=120)
    description: constr(min_length=20, max_length=2000)
    is_public: bool = False

class CoreInsightBase(BaseModel):
    title: constr(max_length=90)
    description: constr(min_length=50, max_length=500)
    sort_order: int = Field(gt=0)

class PersonaBase(BaseModel):
    name: constr(max_length=40)
    role: constr(max_length=60)
    description: constr(min_length=50, max_length=500)
    avatar_url: Optional[str]
    system_prompt: constr(min_length=100, max_length=2000)

class DiscussionBase(BaseModel):
    title: constr(max_length=120)
    sequence_order: int = Field(gt=0)

class MessageBase(BaseModel):
    content: str
    is_user_message: bool
    message_metadata: Optional[dict] = None

# Create schemas (for POST requests)
class UserCreate(UserBase):
    password: str

class CaseCreate(CaseBase):
    file_name: str
    file_path: str
    file_hash: bytes
    file_size: int = Field(gt=0)

class CoreInsightCreate(CoreInsightBase):
    case_id: int

class PersonaCreate(PersonaBase):
    started_case_id: int

class DiscussionCreate(DiscussionBase):
    started_case_id: int

class MessageCreate(MessageBase):
    discussion_id: int
    persona_id: Optional[int]
    user_id: Optional[int]

# Read schemas (for responses)
class User(UserBase):
    user_id: int
    created_at: datetime
    last_active: Optional[datetime]

    class Config:
        from_attributes = True

class Case(CaseBase):
    case_id: int
    file_name: str
    file_path: str
    file_hash: bytes
    file_size: int
    uploader_id: int
    uploaded_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

class CoreInsight(CoreInsightBase):
    insight_id: int
    case_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class StartedCase(BaseModel):
    started_case_id: int
    user_id: int
    case_id: int
    status: CaseStatus
    started_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True

class Persona(PersonaBase):
    persona_id: int
    started_case_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class Discussion(DiscussionBase):
    discussion_id: int
    started_case_id: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

class CoveredInsight(BaseModel):
    covered_insight_id: int
    started_case_id: int
    insight_id: int
    discussion_id: Optional[int]
    covered_at: datetime
    covered_by: Optional[int]
    notes: Optional[str]

    class Config:
        from_attributes = True

class Message(MessageBase):
    message_id: int
    discussion_id: int
    persona_id: Optional[int]
    user_id: Optional[int]
    sent_at: datetime
    read_at: Optional[datetime]

    class Config:
        from_attributes = True
