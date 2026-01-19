from typing import Optional

import numpy as np


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

        self.name_to_index = {name: i for i, name in enumerate(self.yaku_name)}
        self.num_yaku = len(self.yaku_name)

    def encode(self, yaku_list: list[str]) -> np.ndarray:
        """Encode yaku list to vector."""
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
