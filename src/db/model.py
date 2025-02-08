from sqlalchemy.orm import declarative_base
from enum import Enum as PyEnum
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, UniqueConstraint, CheckConstraint, Enum
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import BYTEA, JSONB

Base = declarative_base()

# Define ENUM for case_status
class CaseStatus(PyEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"

# Users table
class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(60), nullable=False)
    password_salt = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_active = Column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("created_at <= COALESCE(last_active, NOW())", name="valid_user_timestamps"),
    )

# Cases table
class Case(Base):
    __tablename__ = "cases"
    
    case_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(120), nullable=False)
    description = Column(Text, nullable=False)
    file_name = Column(String(255), unique=True, nullable=False)
    file_path = Column(String(512), unique=True, nullable=False)
    file_hash = Column(BYTEA, nullable=False)
    file_size = Column(Integer, nullable=False)
    is_public = Column(Boolean, nullable=False, default=False)
    uploader_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("LENGTH(title) >= 5", name="check_title_length"),
        CheckConstraint("LENGTH(description) BETWEEN 20 AND 2000", name="check_description_length"),
        CheckConstraint("file_size > 0", name="check_file_size"),
    )

# Core Insights table
class CoreInsight(Base):
    __tablename__ = "core_insights"
    
    insight_id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.case_id"), nullable=False)
    title = Column(String(90), nullable=False)
    description = Column(Text, nullable=False)
    sort_order = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("case_id", "sort_order", name="unique_case_insight_order"),
        CheckConstraint("LENGTH(description) BETWEEN 50 AND 500", name="check_description_length"),
        CheckConstraint("sort_order > 0", name="check_sort_order"),
    )

# Started Cases table
class StartedCase(Base):
    __tablename__ = "started_cases"
    
    started_case_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    case_id = Column(Integer, ForeignKey("cases.case_id"), nullable=False)
    status = Column(Enum(CaseStatus), nullable=False, default=CaseStatus.NOT_STARTED)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("user_id", "case_id", name="unique_user_case"),
        CheckConstraint("started_at <= COALESCE(completed_at, NOW())", name="valid_case_timeline"),
    )

# Personas table
class Persona(Base):
    __tablename__ = "personas"
    
    persona_id = Column(Integer, primary_key=True, autoincrement=True)
    started_case_id = Column(Integer, ForeignKey("started_cases.started_case_id"), nullable=False)
    name = Column(String(40), nullable=False)
    role = Column(String(60), nullable=False)
    description = Column(Text, nullable=False)
    avatar_url = Column(String(255))
    system_prompt = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("LENGTH(description) BETWEEN 50 AND 500", name="check_description_length"),
        CheckConstraint("LENGTH(system_prompt) BETWEEN 100 AND 2000", name="check_system_prompt_length"),
    )

# Discussions table
class Discussion(Base):
    __tablename__ = "discussions"
    
    discussion_id = Column(Integer, primary_key=True, autoincrement=True)
    started_case_id = Column(Integer, ForeignKey("started_cases.started_case_id"), nullable=False)
    title = Column(String(120), nullable=False)
    sequence_order = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("started_case_id", "sequence_order", name="unique_discussion_order"),
        CheckConstraint("sequence_order > 0", name="check_sequence_order"),
    )

# Covered Insights table
class CoveredInsight(Base):
    __tablename__ = "covered_insights"
    
    covered_insight_id = Column(Integer, primary_key=True, autoincrement=True)
    started_case_id = Column(Integer, ForeignKey("started_cases.started_case_id"), nullable=False)
    insight_id = Column(Integer, ForeignKey("core_insights.insight_id"), nullable=False)
    discussion_id = Column(Integer, ForeignKey("discussions.discussion_id"))
    covered_at = Column(DateTime(timezone=True), server_default=func.now())
    covered_by = Column(Integer, ForeignKey("users.user_id"))
    notes = Column(Text)

    __table_args__ = (
        UniqueConstraint("started_case_id", "insight_id", name="unique_covered_insight"),
    )

# Messages table
class Message(Base):
    __tablename__ = "messages"
    
    message_id = Column(Integer, primary_key=True, autoincrement=True)
    discussion_id = Column(Integer, ForeignKey("discussions.discussion_id"), nullable=False)
    persona_id = Column(Integer, ForeignKey("personas.persona_id"))
    user_id = Column(Integer, ForeignKey("users.user_id"))
    content = Column(Text, nullable=False)
    is_user_message = Column(Boolean, nullable=False)
    sent_at = Column(DateTime(timezone=True), server_default=func.now())
    read_at = Column(DateTime(timezone=True))
    message_metadata = Column(JSONB)  # Renamed from 'metadata' to 'message_metadata'

    __table_args__ = (
        CheckConstraint(
            "(is_user_message = TRUE AND user_id IS NOT NULL AND persona_id IS NULL) OR "
            "(is_user_message = FALSE AND persona_id IS NOT NULL AND user_id IS NULL)",
            name="valid_message_origin"
        ),
        CheckConstraint("sent_at <= COALESCE(read_at, NOW())", name="valid_read_timestamp"),
    )