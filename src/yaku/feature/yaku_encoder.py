import logging
from typing import Optional

import numpy as np
from sqlalchemy.exc import SQLAlchemyError

from src.db.session import get_db_session
from src.db.yaku import Yaku, YakuRepository


class YakuEncoder:
    """Encoder for mapping mahjong yaku to indices."""

    def __init__(self):
        """Initialize yaku encoder."""
        try:
            with get_db_session() as session:
                yaku_repo = YakuRepository(session)
                yaku_list = yaku_repo.find(Yaku.is_yakuman.is_(False), order_by=[Yaku.id.asc()])
                self.yaku_name = [yaku.name for yaku in yaku_list]

        except SQLAlchemyError:
            logging.error("Failed to initialize yaku names from database. Halting.")
            raise

        self.name_to_index = {name: i for i, name in enumerate(self.yaku_name)}
        self.num_yaku = len(self.yaku_name)

    def encode(self, yaku_list: list[str]) -> np.ndarray:
        """Encode yaku list to one-hot vector."""
        vector = np.zeros(self.num_yaku, dtype=np.float32)

        for name in yaku_list:
            if name in self.name_to_index:
                index = self.name_to_index[name]
                vector[index] = 1.0

        return vector

    def get_name(self, index: int) -> Optional[str]:
        """Get yaku name from index."""
        if 0 <= index < self.num_yaku:
            return self.yaku_name[index]

        return None
