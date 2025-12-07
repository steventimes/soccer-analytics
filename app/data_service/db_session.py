import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from app.data_service.db.database.db_schema import Base
from app.data_service.db.data_service import DataService

load_dotenv()
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    logger.warning("DATABASE_URL not set. Using SQLite fallback.")
    DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(engine)

@contextmanager
def get_db_service():
    """
    Context manager that provides a DataService instance with an active session.
    Automatically handles commit/rollback and closing the session.
    
    Usage:
        with get_db_service() as service:
            matches = service.matches.get_by_competition(...)
    """
    session = SessionLocal()
    try:
        service = DataService(session)
        yield service
    except Exception as e:
        logger.error(f"Database Session Error: {e}")
        session.rollback()
        raise e
    finally:
        session.close()