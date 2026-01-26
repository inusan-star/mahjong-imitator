import duckdb
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Union

import src.config as global_config
from src.yaku.common import config as common_config


class YakuEncoder:
    """Encoder for mahjong yaku names."""

    def __init__(self) -> None:
        """Initialize the encoder."""
        if not common_config.YAKU_DISTRIBUTION_FILE.exists():
            raise FileNotFoundError(f"Distribution file not found: {common_config.YAKU_DISTRIBUTION_FILE}")

        yaku_distribution_df = pd.read_csv(common_config.YAKU_DISTRIBUTION_FILE)
        yaku_distribution_df = yaku_distribution_df[yaku_distribution_df["Yaku Name"].notna()]

        self.yaku_names = yaku_distribution_df["Yaku Name"].tolist()
        self.name_to_index = {name: i for i, name in enumerate(self.yaku_names)}
        self.index_to_name = {i: name for i, name in enumerate(self.yaku_names)}
        self.num_classes = len(self.yaku_names)
        self.name_to_mjx_index = self._load_mjx_mapping_with_duckdb()

    def _load_mjx_mapping_with_duckdb(self) -> Dict[str, int]:
        """Load yaku name to MJX index mapping."""
        yaku_parquet_path = f"{global_config.DUMPS_DIR}/yaku/*.parquet"

        connection = duckdb.connect()
        query = f"SELECT id, name FROM read_parquet('{yaku_parquet_path}')"
        yaku_master_df = connection.execute(query).df()
        connection.close()

        return {row.name: int(row.id) - 1 for row in yaku_master_df.itertuples(index=False)}

    def encode(self, yaku_list: List[str]) -> torch.Tensor:
        """Encode yaku names into a multi-hot vector."""
        multi_hot_vector = torch.zeros(self.num_classes)

        for yaku_name in yaku_list:
            if yaku_name in self.name_to_index:
                multi_hot_vector[self.name_to_index[yaku_name]] = 1.0

        return multi_hot_vector

    def decode(self, vector: Union[torch.Tensor, np.ndarray], threshold: float = 0.5) -> List[str]:
        """Decode a vector into yaku names."""
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()

        detected_indices = np.where(vector >= threshold)[0]

        return [self.index_to_name[int(index)] for index in detected_indices]

    def to_mjx_indices(self, yaku_list: List[str]) -> List[int]:
        """Convert yaku names to MJX indices."""
        return [self.name_to_mjx_index[name] for name in yaku_list if name in self.name_to_mjx_index]

    @property
    def labels(self) -> List[str]:
        """Get yaku names."""
        return self.yaku_names
