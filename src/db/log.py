from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session, relationship

from src.db import Base


class Log(Base):
    """Log table model."""

    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String(50), nullable=False, unique=True)
    played_at = Column(DateTime, nullable=False)
    mjlog_status = Column(Integer, nullable=False, default=0)
    mjlog_file_path = Column(String(255), nullable=True, default=None)

    game = relationship("Game", back_populates="log", uselist=False)


class LogRepository:
    """Log repository."""

    def __init__(self, session: Session):
        self._model = Log
        self._session = session

    def find(self, *filters) -> list[Log]:
        """Find."""
        query = self._session.query(self._model)

        if filters:
            query = query.filter(*filters)

        return query.all()

    def bulk_insert(self, records: list[dict]) -> int:
        """Bulk insert."""
        if not records:
            return 0

        statement = mysql_insert(self._model).values(records).prefix_with("IGNORE")
        result = self._session.execute(statement)
        return result.rowcount

    def bulk_update(self, updates: list[dict]):
        """Bulk update."""
        if not updates:
            return

        self._session.bulk_update_mappings(self._model, updates)
