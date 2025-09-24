from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session

from src.db import Base


class Game(Base):
    """Game table model."""

    __tablename__ = "games"

    id = Column(Integer, primary_key=True, autoincrement=True)
    log_url = Column(String(100), nullable=False, unique=True)
    played_at = Column(DateTime, nullable=False)


class GameRepository:
    """Game repository."""

    def __init__(self, session: Session):
        self._model = Game
        self._session = session

    def bulk_insert(self, records: list[dict]) -> int:
        """Bulk insert."""
        if not records:
            return 0

        statement = mysql_insert(self._model).values(records).prefix_with("IGNORE")
        result = self._session.execute(statement)
        return result.rowcount
