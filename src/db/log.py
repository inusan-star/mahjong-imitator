from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session, relationship
from tqdm.rich import tqdm

import src.config as config
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
        self._batch_size = config.DB_BATCH_SIZE

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

        total_inserted = 0

        with tqdm(total=len(records), desc="Inserting LOG", unit="rec") as progress_bar:
            for i in range(0, len(records), self._batch_size):
                batch_records = records[i : i + self._batch_size]
                statement = mysql_insert(self._model).values(batch_records).prefix_with("IGNORE")
                result = self._session.execute(statement)
                total_inserted += result.rowcount
                progress_bar.update(len(batch_records))

        return total_inserted

    def bulk_update(self, updates: list[dict]):
        """Bulk update."""
        if not updates:
            return

        with tqdm(total=len(updates), desc="Updating LOG", unit="rec") as progress_bar:
            for i in range(0, len(updates), self._batch_size):
                batch_records = updates[i : i + self._batch_size]
                self._session.bulk_update_mappings(self._model, batch_records)
                progress_bar.update(len(batch_records))
