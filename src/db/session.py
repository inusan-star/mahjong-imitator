import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


def get_db_engine():
    """Create database engine."""
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    host = os.environ.get("DB_HOST", "db")
    database = os.environ.get("DB_DATABASE")

    return create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")


engine = get_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session() -> Session:
    """Provide database session."""
    session = SessionLocal()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
