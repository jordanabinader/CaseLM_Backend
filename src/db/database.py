import asyncpg
from fastapi import FastAPI

async def get_db_pool(app):
    if not hasattr(app.state, "pool"):
        app.state.pool = await asyncpg.create_pool(
            "postgresql://postgres.yzaovyzvavjdglfzfdfy:hackathon123@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
        )
    return app.state.pool

async def create_message(app, data: dict):
    pool = await get_db_pool(app)
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO messages (
                started_case_id, persona_id, content,
                is_human, awaiting_user_input
            ) VALUES ($1, $2, $3, $4, $5)
        """, data["started_case_id"], data['persona_id'],
            data["content"],
            data["is_user_message"], data["awaiting_user_input"])
    return {"status": "success"}