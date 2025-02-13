# main.py
import uuid
import json
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.workflow.case_discussion_workflow import CaseDiscussionWorkflow
import json
import asyncpg
from src.db.database import get_db_pool
from src.api.endpoints.websocket import router as websocket_router
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.include_router(websocket_router, tags=["websocket"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")

app.mount("/static", StaticFiles(directory=static_dir), name="static")


class SimulationInput(BaseModel):
    case_content: str
    human_participant: Dict[str, Any]


class UserResponse(BaseModel):
    session_id: str
    response: str


class HumanInputRequest(BaseModel):
    session_id: str
    agent_message: str
    input_type: str
    options: Optional[List[str]] = None


# Store active sessions
active_sessions: Dict[str, Dict[str, Any]] = {}


@app.post("/start-discussion")
async def start_discussion(request: Request, data: dict):

    case_content = data.get("case_content", "")
    user_profile = data.get("user_profile", "")
    user_profile = json.loads(user_profile)
    human_participant = user_profile.get("human_participant", "")
    case_id = data.get("case_id", "")

    try:
        # Parse the form data
        input_data = SimulationInput(
            case_content=case_content, human_participant=human_participant
        )

        pool = await get_db_pool(app)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO started_cases (
                    case_id, status
                ) VALUES ($1, $2)
                """,
                case_id,
                "in_progress",
            )

        # Create a new workflow instance
        workflow = CaseDiscussionWorkflow()
        started_case_id = workflow.started_case_id

        print("2")

        session_id = str(uuid.uuid4())

        # Start the workflow
        state = await workflow.run(
            case_content=input_data.case_content,
            human_participant=input_data.human_participant,
        )

        print("3")

        # Store the workflow instance and state
        active_sessions[session_id] = {"workflow": workflow, "state": state}

        # If we're awaiting user input, return the prompt
        if state.get("awaiting_user_input"):
            return {
                "status": "awaiting_input",
                "started_case_id": started_case_id,
                "message": (
                    state.get("messages", [])[-1]["content"]
                    if state.get("messages")
                    else "Your response?"
                ),
            }

        # Otherwise return the complete state
        return {
            "status": "complete",
            "started_case_id": started_case_id,
            "result": state,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit-response")
async def submit_response(request: Request, data: dict):

    response: UserResponse = data.get("response", None)
    try:
        session = active_sessions.get(response.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        workflow = session["workflow"]
        current_state = session["state"]

        # Validate that we're actually waiting for input
        if not current_state.get("awaiting_user_input"):
            raise HTTPException(
                status_code=400, detail="Session is not waiting for user input"
            )

        # Update the state with user response
        current_state["user_response"] = response.response

        # Continue the workflow from the current state
        async for new_state in workflow.graph.astream(current_state):
            # Update session state
            active_sessions[response.session_id]["state"] = new_state

            # If we're awaiting more user input, return the prompt
            if new_state.get("awaiting_user_input"):
                return {
                    "status": "awaiting_input",
                    "session_id": response.session_id,
                    "message": (
                        new_state.get("messages", [])[-1]["content"]
                        if new_state.get("messages")
                        else "Your response?"
                    ),
                    "input_type": new_state.get("input_type", "text"),
                    "options": new_state.get("options"),
                }

            # If workflow is complete, clean up the session
            if new_state.get("complete"):
                del active_sessions[response.session_id]
                return {"status": "complete", "result": new_state}

        # Return the final state if we exit the loop
        return {
            "status": "processing",
            "session_id": response.session_id,
            "state": new_state,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/request-human-input")
async def request_human_input(request: Request, data: dict):

    try:
        session = active_sessions.get(data["session_id"])
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        session["state"]["awaiting_user_input"] = True
        session["state"]["messages"].append(
            {"role": "agent", "content": data["agent_message"]}
        )

        return {
            "status": "awaiting_input",
            "session_id": data["session_id"],
            "message": data["agent_message"],
            "input_type": data["sessioinput_typen_id"],
            "options": data["options"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
async def get_session_status(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "status": (
            "awaiting_input"
            if session["state"].get("awaiting_user_input")
            else "processing"
        ),
        "state": session["state"],
    }


async def create_personas(data: Dict[str, Any]):
    pool = await get_db_pool(app)
    async with pool.acquire() as conn:
        for persona in data["personas"]:
            await conn.execute(
                """
                INSERT INTO personas (
                    started_case_id, persona_id, name, role, background, personality,
                    expertise, is_human, voice
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                data["started_case_id"],
                persona["uuid"],
                persona["name"],
                persona["role"],
                persona["background"],
                persona["personality"],
                persona["expertise"],
                persona["is_human"],
                persona["voice"],
            )
    return {"status": "success"}


async def create_topics(data: Dict[str, Any]):
    pool = await get_db_pool(app)
    async with pool.acquire() as conn:
        for idx, topic in enumerate(data["topics"], 1):
            await conn.execute(
                """
                INSERT INTO topics (
                    started_case_id, title, expected_insights, topic_id
                ) VALUES ($1, $2, $3, $4)
            """,
                data["started_case_id"],
                topic["title"],
                topic["expected_insight"],
                idx,
            )
    return {"status": "success"}


async def create_message(data: Dict[str, Any]):
    print(f"data: {data}")
    pool = await get_db_pool(app)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO messages (
                started_case_id, persona_id, content,
                is_human, awaiting_user_input
            ) VALUES ($1, $2, $3, $4, $5)
        """,
            data["started_case_id"],
            data["persona_id"],
            data["content"],
            data["is_user_message"],
            data["awaiting_user_input"],
        )
    return {"status": "success"}


async def get_unread_messages(started_case_id: int):
    pool = await get_db_pool(app)
    async with pool.acquire() as conn:
        messages = await conn.fetch(
            """
            SELECT content
            FROM messages 
            WHERE started_case_id = $1 
            AND is_human IS TRUE
            ORDER BY time_sent ASC
        """,
            started_case_id,
        )

        return [dict(msg) for msg in messages]


@app.get("/health")
async def health_check():
    try:
        # Get database pool
        pool = await get_db_pool(app)

        # Test database connection with a simple query
        async with pool.acquire() as conn:
            # Check if we can execute a simple query
            db_result = await conn.fetchval("SELECT NOW()")

            # Get some basic stats
            case_count = await conn.fetchval("SELECT COUNT(*) FROM cases")
            message_count = await conn.fetchval("SELECT COUNT(*) FROM messages")

            return {
                "status": "healthy",
                "database": {
                    "connected": True,
                    "timestamp": db_result,
                    "stats": {
                        "total_cases": case_count,
                        "total_messages": message_count,
                    },
                },
            }

    except Exception as e:
        return {
            "status": "unhealthy",
            "database": {"connected": False, "error": str(e)},
        }


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "test.html"))


@app.get("/test-stt")
async def test_stt():
    return FileResponse("src/static/test.html")


# If running directly, start the FastAPI server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
