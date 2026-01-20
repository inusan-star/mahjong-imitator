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
        self.yaku_name = [
            "立直",
            "断幺九",
            "場風 東",
            "場風 北",
            "役牌 白",
            "役牌 發",
            "役牌 中",
            "混一色",
            "ドラ",
        ]

        name_to_idx = {name: i for i, name in enumerate(self.yaku_name)}
        self.id_to_index = {}

        try:
            with get_db_session() as session:
                yaku_repo = YakuRepository(session)
                all_yaku = yaku_repo.find()

                for yaku in all_yaku:
                    if yaku.name in name_to_idx:
                        self.id_to_index[yaku.id] = name_to_idx[yaku.name]

        except SQLAlchemyError:
            logging.error("Failed to initialize yaku mapping from database.")
            raise

        self.num_yaku = len(self.yaku_name)

        logging.info("Initialized YakuEncoder with %d yaku IDs from database.", len(self.id_to_index))

    def encode(self, yaku_ids: list[int]) -> np.ndarray:
        """Encode yaku list to vector."""
        vector = np.zeros(self.num_yaku, dtype=np.float32)

        for yaku_id in yaku_ids:
            if yaku_id in self.id_to_index:
                index = self.id_to_index[yaku_id]
                vector[index] = 1.0

        return vector

    def get_name(self, index: int) -> Optional[str]:
        """Get yaku name from index."""
        if 0 <= index < self.num_yaku:
            return self.yaku_name[index]

        return None
