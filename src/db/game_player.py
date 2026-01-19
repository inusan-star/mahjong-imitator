from sqlalchemy import Column, Float, ForeignKey, Integer
from sqlalchemy.orm import relationship, Session
from typing import Any, Optional

from src.db import Base


class GamePlayer(Base):
    """GamePlayer table model."""

    __tablename__ = "game_players"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    seat_index = Column(Integer, nullable=False)
    dan = Column(Integer, nullable=False)
    rate = Column(Float, nullable=False)
    score = Column(Integer, nullable=False)
    rank = Column(Integer, nullable=False)

    game = relationship("Game")
    player = relationship("Player")


class GamePlayerRepository:
    """GamePlayer repository."""

    def __init__(self, session: Session):
        self._model = GamePlayer
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
