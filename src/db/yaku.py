from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import Session
from typing import Any, Optional

from src.db import Base


class Yaku(Base):
    """Yaku table model."""

    __tablename__ = "yaku"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False, unique=True)
    han_closed = Column(Integer, nullable=False)
    han_open = Column(Integer, nullable=False)
    is_yakuman = Column(Boolean, nullable=False)


class YakuRepository:
    """Yaku repository."""

    def __init__(self, session: Session):
        self._model = Yaku
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
