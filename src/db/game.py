from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from src.db import Base


class Game(Base):
    """Game table model."""

    __tablename__ = "games"

    id = Column(Integer, primary_key=True, autoincrement=True)
    log_id = Column(Integer, ForeignKey("logs.id"), nullable=False, unique=True)

    log = relationship("Log", back_populates="game")
