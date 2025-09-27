from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for models."""


from src.db.game import Game
from src.db.log import Log
