from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Mapped, mapped_column, Mapper, Session
from typing import Any, cast, Optional

from src.db import Base


class Player(Base):
    """Player table model."""

    __tablename__ = "players"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    gender = Column(String(10), nullable=False)
    last_dan = Column(Integer, nullable=False)
    last_rate = Column(Float, nullable=False)
    max_dan = Column(Integer, nullable=False)
    max_rate = Column(Float, nullable=False)
    last_played_at = Column(DateTime, nullable=False)
    game_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    second_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    third_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fourth_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class PlayerRepository:
    """Player repository."""

    def __init__(self, session: Session):
        self._model = Player
        self._session = session

    def find(
        self,
        *filters,
        order_by: Optional[list[Any]] = None,
        outer_joins: Optional[list[Any]] = None,
    ):
        """Find."""
        query = self._session.query(self._model)

        if outer_joins:
            for join_table in outer_joins:
                query = query.outerjoin(join_table)

        if filters:
            query = query.filter(*filters)

        if order_by:
            query = query.order_by(*order_by)

        return query.all()

    def bulk_insert(self, records: list[dict]):
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

        self._session.bulk_update_mappings(cast(Mapper[Any], self._model), updates)
