import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from app.data_service.db.database.db_schema import Base
from app.data_service.db.data_service import DataService

#test if docker has correctly set up the env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    print("DATABASE_URL environment variable is not set. Using a default SQLite database.")
    DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
session_local = sessionmaker(bind=engine)
Base.metadata.create_all(engine)


def get(type, id):
    session = session_local()
    try:
        service = DataService(session)
        return service.data_get(type, id)
    finally:
        session.close()
