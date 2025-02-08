import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from model import Base  # Corrected import path
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Use environment variable for the database URL
DATABASE_URL = "postgresql+psycopg2://postgres:your_password@localhost/caselm_db"
print(DATABASE_URL)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()