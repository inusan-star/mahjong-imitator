from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from src.db import Base


class RoundYaku(Base):
    """RoundYaku table model."""

    __tablename__ = "round_yaku"

    id = Column(Integer, primary_key=True, autoincrement=True)
    round_id = Column(Integer, ForeignKey("rounds.id"), nullable=False)
    yaku_id = Column(Integer, ForeignKey("yaku.id"), nullable=False)

    round = relationship("Round")
    yaku = relationship("Yaku")
